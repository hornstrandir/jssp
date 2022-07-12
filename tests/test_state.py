import pytest
import numpy as np
from env.jss_env_dir.jss_env import JssEnv

@pytest.fixture
def env():
    """By default, JssEnv uses Ta01 unless otherwise specified."""
    return JssEnv()

def test_initial_state(env):
    test_env = env
    test_state = env.reset()
    assert test_env.current_time_step == 0
    assert max(test_state["real_obs"].flatten()) <= 1.0
    assert min(test_state["real_obs"].flatten()) >= 0.0
    assert not np.isnan(test_state["real_obs"]).any()
    assert not np.isinf(test_state["real_obs"]).any()

def test_nb_machines_available(env):
    for _ in range(100):
        env.reset()
        assert env.current_time_step == 0
        machines_available = set()
        for job in range(len(env.legal_actions[:-1])):
            if env.legal_actions[job]:
                machine_needed = env.needed_machine_jobs[job]
                machines_available.add(machine_needed)
        assert len(machines_available) == env.nb_machine_legal
        
def test_random_actions(env):
    average = 0 
    for _ in range(100):
        env.reset()
        legal_actions = env.get_legal_actions()
        done = False
        total_reward = 0
        while not done:
            actions = np.random.choice(
                len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
                )[0]
            assert legal_actions[:-1].sum() == env.nb_legal_actions
            state, rewards, done, _ = env.step(actions)
            legal_actions = env.get_legal_actions()
            total_reward += rewards
            assert max(state["real_obs"].flatten()) <= 1.0, "Out of max bound state"
            assert min(state["real_obs"].flatten()) >= 0.0, "Out of min bound state"
            assert not np.isnan(state["real_obs"]).any(), "NaN inside state rep!"
            assert not np.isinf(state["real_obs"]).any(), "Inf inside state rep!"
            machines_available = set()
            for job in range(len(env.legal_actions[:-1])):
                if env.legal_actions[job]:
                    machine_needed = env.needed_machine_jobs[job]
                    machines_available.add(machine_needed)
            assert len(machines_available) == env.nb_machine_legal
        average += env.last_time_step
        assert len(env.next_time_step) == 0
        assert min(env.solution.flatten()) != -1
        #assert min(env.solution.flatten()) != np.array2string(env.solution.flatten())
        for job in range(env.jobs):
            assert env.todo_time_step_job[job] == env.machines

