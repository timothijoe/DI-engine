import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import DQNPolicy
from ding.policy import EfficientZeroPolicy
# from ding.policy.metadrivezero import MetadriveZeroPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, AdvancedReplayBuffer
from ding.rl_utils import get_epsilon_greedy_fn
from core.envs import DriveEnvWrapper, MetaDriveMacroEnv

from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
import numpy as np
from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
#from ding.worker import EpisodeSerialCollectorMuZero
from ding.worker.collector.metadrive_collector import MetadriveCollector
from ding.worker.collector.base_serial_evaluator_muzero import BaseSerialEvaluatorMuZero as BaseSerialEvaluator

from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.data.buffer.game_buffer import GameBuffer
from ding.data.buffer.metadrive_buffer import MetadriveBuffer
from dizoo.board_games.atari.config.metadrive_config import game_config

metadrive_macro_config = dict(
    exp_name='mcts_data/data_ez_ptree/mcts_tree',
    env=dict(
        metadrive=dict(use_render=False),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=2,
        stop_value=99999,
        collector_env_num=2,
        evaluator_env_num=1,
        wrapper=dict(),
        max_episode_steps = int(150),
        gray_scale = False,
        cvt_string=True,
        game_wrapper = True,
        dqn_expert_data = False, 
    ),
    policy=dict(
        cuda=False,
        env_name='PongNoFrameskip-v4',
        model=dict(
            model_type='atari',
            observation_shape=(5, 200, 200),  # 3,96,96 stack=4
            action_space_size=5,
            downsample=True,
            num_blocks=1,
            num_channels=64,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            lstm_hidden_size=512,
            bn_mt=0.1,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            init_zero=True,
            state_norm=False,   
        ),
        learn=dict(
            # debug
            update_per_collect=8,
            batch_size=4,
            momentum = 0.9,
            weight_decay = 0.0001,

            # update_per_collect=200,  # TODO(pu): 1000
            # batch_size=256,

            learning_rate=0.2,
            # Frequency of target network update.
            target_update_freq=400,
        ),
        collect=dict(
            n_sample=50,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(
                replay_buffer_size=10000,
                type='game',
            ),
        ),
    ),
)

main_config = EasyDict(metadrive_macro_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveMacroEnv(env_cfg), wrapper_cfg)

def main(cfg):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True,
    )
    print(cfg.policy.collect.collector)

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )
    total_field = ['learn', 'collect', 'eval']
    policy = EfficientZeroPolicy(cfg.policy,enable_field=total_field)
    print('zt')
    print('zt')
    print('zt')
    print('zt')
    print('zt')
    print('zt')
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    # collector = SampleSerialCollector(
    #     cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    # )
    # evaluator = InteractionSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    # )
    # replay_buffer = GameBuffer(game_config)
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # MuZero related code
    # specific game buffer for MuZero
    replay_buffer = MetadriveBuffer(game_config)
    # collector = SampleSerialCollector(
    #     cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    # )
    collector = MetadriveCollector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        replay_buffer=replay_buffer,
        game_config=game_config
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator,
        evaluator_env,
        policy.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        game_config=game_config
    )
    # collect_kwargs['temperature'] = np.array(
    #     [
    #         game_config.visit_count_temperature(trained_steps=learner.train_iter)
    #         for _ in range(game_config.collector_env_num)
    #     ]
    # )
    

    new_data = collector.collect(n_episode=5, train_iter=learner.train_iter)
    replay_buffer.remove_to_fit()
    train_data = replay_buffer.sample_train_data(learner.policy.get_attribute('batch_size'), policy)
    learner.train(train_data, collector.envstep)

    # commander = BaseSerialCommander(
    #     cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    # )
    
    
    
    
    print('zt2')
    print('zt2')
    
    
    
    
    # if evaluator.should_eval(learner.train_iter):
    #     stop, reward = evaluator.eval(
    #     learner.save_checkpoint, learner.train_iter, collector.envstep, config=game_config
    # )
    # if stop:
    #     break   
if __name__ == '__main__':
    main(main_config)