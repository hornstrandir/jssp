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
        self.legal_actions = None # (a1) boolean to indicate if job can be allocated
        self.time_until_available_machine = None # (a5) required time until machine is free 
        self.time_until_finish_current_op_jobs = None # (a2) left-over time for currently performed operation on the job
        self.todo_time_step_job = None # (a4) left-over time until completion of job
        self.total_perform_op_time_jobs = None # (a3) percentage of op finished for a job
        self.needed_machine_jobs = None # 
        self.total_idle_time_jobs = None #(a7) cumulative job's idle time in the schedule
        self.idle_time_jobs_last_op = None # (a6) Idle time since last job's performed operation
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
        self.state[:,0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions, 
        }

    def get_legal_actions(self):
        return self.legal_actions

    def reset(self):
        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all legal actions
        self.legal_actions = np.ones(self.jobs+1, dtype=bool)
        self.legal_actions[self.jobs] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=int)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.jobs, dtype=bool)
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        for nb_job in range(self.jobs):
            needed_machine = self.instance_matrix[nb_job][0][0]
            self.needed_machine_jobs[nb_job] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
                self.nb_machine_legal += 1
        self.state = np.zeros((self.jobs, 7), dtype=float)
        return self._get_current_state_representation()


    def _priorization_non_final(self):
        """
        Set legal action of final job to False and reduce the number of legal actions 
        by one, if there is a non-final job that can be allocated to the same machine.
        """
        if self.nb_machine_legal >= 1:
            for machine in range(self.machines):
                if self.machine_legal[machine]:
                    final_job = list()
                    non_final_job = list()
                    min_non_final = float("inf")
                    for job in range(self.jobs):
                        if (
                            self.needed_machine_jobs[job] == machine 
                            and self.legal_actions[job]
                        ):
                            if self.todo_time_step_job[job] == (self.machines - 1):
                                final_job.append(job)
                            else:
                                current_time_step_non_final = self.todo_time_step_job[
                                    job
                                ]
                                time_needed_legal = self.instance_matrix[job][
                                    current_time_step_non_final
                                ][1]
                                machine_needed_next_step = self.instance_matrix[job][
                                    current_time_step_non_final + 1
                                ][0]
                                if (
                                    self.time_until_available_machine[
                                        machine_needed_next_step
                                    ]
                                    == 0
                                ):
                                    min_non_final = min(
                                        min_non_final, time_needed_legal
                                    )
                                    non_final_job.append(job)
            if len(non_final_job) > 0:
                for job in final_job:
                    current_time_step_final = self.todo_time_step_job[job]
                    time_needed_legal = self.instance_matrix[job][
                        current_time_step_final
                        ][1]
                    if time_needed_legal > min_non_final:
                        self.legal_actions[job] = False
                        self.nb_legal_actions -= 1

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