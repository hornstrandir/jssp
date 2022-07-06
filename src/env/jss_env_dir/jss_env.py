import gym
import numpy as np

class JssEnv(gym.Env):

    def __init__(self, env_config=None) -> None:
        """Docstring"""
        if env_config is None:
            env_config = {
                "instance_path": str(Path(__file__).parent.absolute())
                + "/instances/ta80"
            }
        instance_path = env_config["instance_path"]

        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0 # longest duration of an operation
        self.max_time_jobs = 0 # longest job total completion time
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        #
        

    def _get_current_state_representation(self):
        pass

    def get_legal_actions(self):
        pass

    def reset(self):
        print('Env reset')

    def priorization_non_final(self):
        pass

    def _check_no_op(self):
        pass

    def step(self):
        print('Step successful')

    def _reward_scaler(self):
        pass

    def increase_time_step(self):
        pass

    def _is_done(self):
        pass

    def render(self):
        pass