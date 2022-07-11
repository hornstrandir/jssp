import gym
import datetime
import numpy as np
from pathlib import Path
import random

class JssEnv(gym.Env):

    def __init__(self, env_config=None) -> None:
        """Docstring"""
        if env_config is None:
            env_config = {
                "instance_path": str(Path(__file__).parent.absolute())
                + "/../instances/ta01"
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
        self.solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        with open(instance_path, "r") as file:
            lines = file.readlines()
            splitted_lines = [line.split() for line in lines]
            self.jobs = int(splitted_lines[0][0])
            # matrix which store tuple of (machine, length of the job)
            self.machines = int(splitted_lines[0][1])
            self.instance_matrix = np.zeros(
                (self.jobs, self.machines), dtype=(int, 2)
            )
            # contains all the time to complete jobs
            self.jobs_length = np.zeros(self.jobs, dtype=int)
            for nb_job, splitted_line in enumerate(splitted_lines[1:]):
                assert len(splitted_line) % 2 == 0
                # each jobs must pass a number of operation equal to the number of machines
                assert len(splitted_line) / 2 == self.machines
                for i in range(0, len(splitted_line), 2):
                    print(i//2)                 
                    machine, time = int(splitted_line[i]), int(splitted_line[i + 1])
                    self.instance_matrix[nb_job][i//2] = (machine, time)
                    self.max_time_op = max(self.max_time_op, time)
                    self.jobs_length[nb_job] += time
                    self.sum_op += time
        self.max_time_jobs = max(self.jobs_length)
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1 #at least 2 machines
        assert self.instance_matrix is not None
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
                # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        """
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float
                ),
            }
        )





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