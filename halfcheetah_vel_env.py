import numpy as np
import gymnasium as gym

class HalfCheetahVelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, task=None, render_mode=None):
        if task is None:
            task = {"velocity": 0.0}
        self._task     = task
        self._goal_vel = task["velocity"]

        self.env = gym.make("HalfCheetah-v4", render_mode=render_mode)

        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        data = self.env.unwrapped.data   # mujoco.MjData
        xpos_before = data.qpos[0]

        obs, _orig_reward, terminated, truncated, info = self.env.step(action)

        xpos_after   = data.qpos[0]
        forward_vel  = (xpos_after - xpos_before) / self.env.unwrapped.dt
        forward_rwd  = -abs(forward_vel - self._goal_vel)
        ctrl_cost    = 0.5 * 1e-1 * np.sum(np.square(action))
        custom_rwd   = forward_rwd - ctrl_cost

        info.update({
            "reward_forward": forward_rwd,
            "reward_ctrl":   -ctrl_cost,
            "task":           self._task
        })

        return obs, custom_rwd, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def sample_tasks(self, num_tasks):
        velocities = self.env.unwrapped.np_random.uniform(0.0, 2.0, size=(num_tasks,))
        return [{"velocity": v} for v in velocities]

    def reset_task(self, task):
        self._task     = task
        self._goal_vel = task["velocity"]
