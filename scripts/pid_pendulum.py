# from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import numpy as np


class PendulumPID(object):

    def __init__(self, Kp, Ki, Kd, target, **kwargs):
        self._env = TfEnv(GymEnv('Pendulum-v0',
                                 record_video=kwargs.get("record_video", False),
                                 record_log=kwargs.get("record_log", False)))
        self._int, self._diff = 0.0, 0.0
        self._Kp, self._Ki, self._Kd, self._target = Kp, Ki, Kd, target
        self._last_obs = self._env.reset()

    @property
    def env(self):
        return self._env

    def step(self):
        L = 1.0
        m = 1.0
        g = 10.0

        theta, theta_dot = np.arccos(self._last_obs[0]), self._last_obs[2] 

        # taw = -L * m * g * np.sin(np.pi - theta)
        # target_taw = -L * m * g * np.sin(np.pi - self._target)

        error = -L * m * g * np.sin(theta - self._target)
        # error = target_taw
        self._int += error
        new_taw = self._Kp * error + self._Ki * self._int + self._Kd * (error - self._diff) 
        
        self._diff = error

        self._last_obs, r, d, info = self._env.step([new_taw])

        info.update({"action": new_taw})

        return self._last_obs, r, d, info