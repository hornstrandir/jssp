from gym.envs.registration import register

register(id='JssEnv-v0',
    entry_point='env.jss_env_dir:JssEnv')