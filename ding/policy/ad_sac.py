from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, q_v_1step_td_error, q_v_1step_td_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from ding.model.template.ad_vae import VAELSTM
from ding.policy.sac import SACPolicy
from ding.model import create_model
from ding.utils import import_module, allreduce, broadcast, get_rank, allreduce_async, synchronize, POLICY_REGISTRY
@POLICY_REGISTRY.register('ad_sac')
class ADSAC(SACPolicy):

    def __init__(
            self,
            cfg: dict,
            model = None,
            enable_field= None
    ) -> None:
        self._cfg = cfg
        self._on_policy = self._cfg.on_policy
        if enable_field is None:
            self._enable_field = self.total_field
        else:
            self._enable_field = enable_field
        assert set(self._enable_field).issubset(self.total_field), self._enable_field

        if len(set(self._enable_field).intersection(set(['learn', 'collect', 'eval']))) > 0:
            model = self._create_model_ad(cfg, model)
            self._cuda = cfg.cuda and torch.cuda.is_available()
            # now only support multi-gpu for only enable learn mode
            if len(set(self._enable_field).intersection(set(['learn']))) > 0:
                self._rank = get_rank() if self._cfg.learn.multi_gpu else 0
                if self._cuda:
                    torch.cuda.set_device(self._rank % torch.cuda.device_count())
                    model.cuda()
                if self._cfg.learn.multi_gpu:
                    bp_update_sync = self._cfg.learn.get('bp_update_sync', True)
                    self._bp_update_sync = bp_update_sync
                    self._init_multi_gpu_setting(model, bp_update_sync)
            else:
                self._rank = 0
                if self._cuda:
                    torch.cuda.set_device(self._rank % torch.cuda.device_count())
                    model.cuda()
            self._model = model
            self._device = 'cuda:{}'.format(self._rank % torch.cuda.device_count()) if self._cuda else 'cpu'
        else:
            self._cuda = False
            self._rank = 0
            self._device = 'cpu'

        for field in self._enable_field:
            getattr(self, '_init_' + field)()
    def _init_learn(self):
        super()._init_learn()
        self._vae_model = VAELSTM()
        model_PATH = '/home/SENSETIME/zhoutong/hoffnung/motion_primitive_vae/result/christmas105/ckpt/47_ckpt'
        self._vae_model.load_state_dict(torch.load(model_PATH))
        self._optimizer_vae = Adam(self._vae_model.parameters(), lr = self._cfg.learn.learning_rate_value)
    def ad_default_model(self) -> Tuple[str, List[str]]:
        return 'ad_qac', ['ding.model.template.ad_qac']
        # if self._cfg.multi_agent:
        #     return 'maqac_continuous', ['ding.model.template.maqac']
        # else:
        #     return 'qac', ['ding.model.template.qac']

    def _create_model_ad(self, cfg: dict, model= None) -> torch.nn.Module:
        if model is None:
            model_cfg = cfg.model
            if 'type' not in model_cfg:
                m_type, import_names = self.ad_default_model()
                model_cfg.type = m_type
                model_cfg.import_names = import_names
            return create_model(model_cfg)
        else:
            if isinstance(model, torch.nn.Module):
                return model
            else:
                raise RuntimeError("invalid model: {}".format(type(model)))
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr, loss, target_q_value and other \
                running information.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']

        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        # 2. predict target value depend self._value_network.
        if self._value_network:
            v_value = self._learn_model.forward(obs, mode='compute_value_critic')['v_value']
            with torch.no_grad():
                next_v_value = self._target_model.forward(next_obs, mode='compute_value_critic')['v_value']
            target_q_value = next_v_value
        else:
            # target q value.
            with torch.no_grad():
                (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

                dist = Independent(Normal(mu, sigma), 1)
                pred = dist.rsample()
                next_action = torch.tanh(pred)
                y = 1 - next_action.pow(2) + 1e-6
                # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
                next_log_prob = dist.log_prob(pred).unsqueeze(-1)
                next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)

                next_data = {'obs': next_obs, 'action': next_action}
                target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
                # the value of a policy according to the maximum entropy objective
                if self._twin_critic:
                    # find min one as target q value
                    target_q_value = torch.min(target_q_value[0],
                                               target_q_value[1]) - self._alpha * next_log_prob.squeeze(-1)
                else:
                    target_q_value = target_q_value - self._alpha * next_log_prob.squeeze(-1)

        # 3. compute q loss
        if self._twin_critic:
            q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done, data['weight'])
            loss_dict['twin_critic_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # 4. update q network
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        if self._twin_critic:
            loss_dict['twin_critic_loss'].backward()
        self._optimizer_q.step()

        # 5. evaluate to get action distribution
        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        y = 1 - action.pow(2) + 1e-6
        # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        # 6. (optional) compute value loss and update value network
        if self._value_network:
            # new_q_value: (bs, ), log_prob: (bs, act_shape) -> target_v_value: (bs, )
            target_v_value = (new_q_value.unsqueeze(-1) - self._alpha * log_prob).mean(dim=-1)
            loss_dict['value_loss'] = F.mse_loss(v_value, target_v_value.detach())

            # update value network
            self._optimizer_value.zero_grad()
            loss_dict['value_loss'].backward()
            self._optimizer_value.step()

        # 7. compute policy loss
        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        loss_dict['policy_loss'] = policy_loss

        # 8. update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # 9. compute alpha loss
        if self._auto_alpha:
            if self._log_space:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
            else:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = max(0, self._alpha)

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_q_value': target_q_value.detach().mean().item(),
            **loss_dict
        }

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        init_state_batch = data[:, 1, 0, 0: 4]
        self._collect_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._collect_model.forward(data, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            action = torch.tanh(dist.rsample())
            output = {'logit': (mu, sigma), 'action': action}
            output['latent_action'] = output['action']
            traj = self._vae_model.decode(output['action'], init_state_batch)
            traj = torch.cat([init_state_batch.unsqueeze(1), traj], dim = 1)
            output['action'] = traj[:, :,:2]
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        init_state_batch = data[:, 1, 1, 0: 4]
        self._eval_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._eval_model.forward(data, mode='compute_actor')['logit']
            action = torch.tanh(mu)  # deterministic_eval
            output = {'latent_action': action}
            output['action'] = self._vae_model.decode(output['latent_action'], init_state_batch)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """

        transition = {
            'obs' : obs,
            'next_obs': timestep.obs,
            'latent_action': model_output['latent_action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition