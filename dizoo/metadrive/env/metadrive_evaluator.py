from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer

from typing import Optional, Callable, Tuple
from collections import namedtuple
import numpy as np
import torch

from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from ding.worker import ISerialEvaluator, VectorEvalMonitor
from typing import Optional, Callable, Tuple
from collections import namedtuple
import numpy as np
import torch
import torch.distributed as dist

from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray, to_item
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from ding.utils import get_world_size, get_rank



@SERIAL_EVALUATOR_REGISTRY.register('meta-interaction')
class MetadriveEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """
    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            force_render: bool = False,
    ) -> Tuple[bool, dict]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - return_info (:obj:`dict`): Current evaluation return information.
        '''
        if get_world_size() > 1:
            # sum up envstep to rank0
            envstep_tensor = torch.tensor(envstep).cuda()
            dist.reduce(envstep_tensor, dst=0)
            envstep = envstep_tensor.item()

        z_success_times = 0 
        z_fail_times = 0 
        z_crash_vehicle_times = 0
        # z_arrive_dest_times = 0
        z_out_of_road_times = 0
        z_timed_out = 0
        complete_ratio_list = []

        # evaluator only work on rank0
        stop_flag, return_info = False, []
        if get_rank() == 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "please indicate eval n_episode"
            envstep_count = 0
            info = {}
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            self._env.reset()
            self._policy.reset()

            # force_render overwrite frequency constraint
            render = force_render or self._should_render(envstep, train_iter)

            with self._timer:
                while not eval_monitor.is_finished():
                    obs = self._env.ready_obs
                    obs = to_tensor(obs, dtype=torch.float32)

                    # update videos
                    if render:
                        eval_monitor.update_video(self._env.ready_imgs)

                    policy_output = self._policy.forward(obs)
                    actions = {i: a['action'] for i, a in policy_output.items()}
                    actions = to_ndarray(actions)
                    timesteps = self._env.step(actions)
                    timesteps = to_tensor(timesteps, dtype=torch.float32)
                    for env_id, t in timesteps.items():
                        if t.info.get('abnormal', False):
                            # If there is an abnormal timestep, reset all the related variables(including this env).
                            self._policy.reset([env_id])
                            continue
                        if t.done:
                            # Env reset is done by env_manager automatically.
                            if 'figure_path' in self._cfg:
                                if self._cfg.figure_path is not None:
                                    self._env.enable_save_figure(env_id, self._cfg.figure_path)
                            self._policy.reset([env_id])
                            reward = t.info['eval_episode_return']

                            arrive_dest = t.info['arrive_dest']
                            if arrive_dest:
                                z_success_times += 1 
                            else:
                                z_fail_times += 1 
                            if t.info['crash_vehicle']:
                                z_crash_vehicle_times += 1
                            if t.info['out_of_road']:
                                z_out_of_road_times += 1 
                            if t.info['crash_building']:
                                z_timed_out += 1 
                            # complete_ratio_list.append(float(t.info['complete_ratio']))
                        
                        
                            
                            if 'complete_ratio' in t.info:
                                complete_ratio_list.append(float(t.info['complete_ratio']))                         
                            
                            
                            
                            if 'episode_info' in t.info:
                                eval_monitor.update_info(env_id, t.info['episode_info'])
                            eval_monitor.update_reward(env_id, reward)
                            return_info.append(t.info)
                            self._logger.info(
                                "[EVALUATOR]env {} finish episode, final reward: {:.4f}, current episode: {}".format(
                                    env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                                )
                            )
                        envstep_count += 1
            duration = self._timer.value
            episode_return = eval_monitor.get_episode_return()
            success_ratio = float(z_success_times) / float(z_success_times + z_fail_times)
            crash_vehicle_ratio = float(z_crash_vehicle_times) / float(z_success_times + z_fail_times)
            out_of_road_ratio = float(z_out_of_road_times) / float(z_success_times + z_fail_times)
            timed_out_ratio = float(z_timed_out)/ float(z_success_times + z_fail_times)
            info = {
                'train_iter': train_iter,
                'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
                'episode_count': n_episode,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / n_episode,
                'evaluate_time': duration,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_time_per_episode': n_episode / duration,
                'reward_mean': np.mean(episode_return),
                'reward_std': np.std(episode_return),
                'reward_max': np.max(episode_return),
                'reward_min': np.min(episode_return),
                # 'each_reward': episode_return,
                'complete_ratio': np.mean(complete_ratio_list),
                'succ_rate': success_ratio,
                'crash_vehicle_ratio': crash_vehicle_ratio,
                'out_of_road_ratio': out_of_road_ratio,
                'timed_out_ratio' : timed_out_ratio,
            }
            episode_info = eval_monitor.get_episode_info()
            if episode_info is not None:
                info.update(episode_info)
            self._logger.info(self._logger.get_tabulate_vars_hor(info))
            # self._logger.info(self._logger.get_tabulate_vars(info))
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward']:
                    continue
                if not np.isscalar(v):
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)

            if render:
                video_title = '{}_{}/'.format(self._instance_name, self._render.mode)
                videos = eval_monitor.get_video()
                render_iter = envstep if self._render.mode == 'envstep' else train_iter
                from ding.utils import fps
                self._tb_logger.add_video(video_title, videos, render_iter, fps(self._env))

            episode_return = np.mean(episode_return)
            if episode_return > self._max_episode_return:
                if save_ckpt_fn:
                    save_ckpt_fn('ckpt_best.pth.tar')
                self._max_episode_return = episode_return
            stop_flag = episode_return >= self._stop_value and train_iter > 0
            if stop_flag:
                self._logger.info(
                    "[DI-engine serial pipeline] " + "Current episode_return: {:.4f} is greater than stop_value: {}".
                    format(episode_return, self._stop_value) + ", so your RL agent is converged, you can refer to " +
                    "'log/evaluator/evaluator_logger.txt' for details."
                )

        if get_world_size() > 1:
            objects = [stop_flag, return_info]
            dist.broadcast_object_list(objects, src=0)
            stop_flag, return_info = objects

        return_info = to_item(return_info)
        return stop_flag, return_info

