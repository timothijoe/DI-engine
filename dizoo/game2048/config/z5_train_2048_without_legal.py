from easydict import EasyDict

game2048_dqn_config = dict(
    exp_name='game2048_dqn_v5',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        obs_shape=(16, 4, 4),
        stop_value=int(1e6),
        channel_last=False,
        reward_scale=100,
        obs_type='dict_observation',  # options=['raw_observation', 'dict_observation']
        ignore_legal_actions=False,
        
    ),
    policy=dict(
        cuda=True,
        priority=True,
        model=dict(
            obs_shape=(16, 4, 4),
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 128, 256],
        ),
        nstep=10,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=100,
            batch_size=128,
            learning_rate=0.003,
            target_update_freq=500,
        ),
        collect=dict(n_sample=1024, ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=int(1e6),
            ),
            replay_buffer=dict(replay_buffer_size=1000000, ),
        ),
    ),
)
game2048_dqn_config = EasyDict(game2048_dqn_config)
main_config = game2048_dqn_config
game2048_dqn_create_config = dict(
    env=dict(
        type='game_2048',
        import_names=['dizoo.game2048.envs.game_2048'],
    ),
    env_manager=dict(type='subprocess', 
                     shared_memory=False),
    policy=dict(type='dqn'),
)
game2048_dqn_create_config = EasyDict(game2048_dqn_create_config)
create_config = game2048_dqn_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c game2048_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)