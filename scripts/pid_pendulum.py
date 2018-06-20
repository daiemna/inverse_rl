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
        self._alpha_dot_prev = self._last_obs[2]
        self._alpha_tol = 0.2


    @property
    def env(self):
        return self._env

    def step(self):
        r = 1.0
        Ip = r/2.0
        m = 1.0
        g = 10.0
        dt = 2.0

        alpha, alpha_dot = np.arccos(self._last_obs[0]), self._last_obs[2]
        alpha_dotdot = (alpha_dot - self._alpha_dot_prev)/dt
        self._alpha_dot_prev = alpha_dot

        taw_stablity = (m * alpha_dotdot * Ip**2) - (m * Ip * np.sin(alpha) * alpha_dot * r) - (m * g * Ip * np.sin(alpha))

        if abs(alpha - self._target) > self._alpha_tol:
            new_taw = -np.sign(alpha_dot) * (taw_stablity)
        else:
            error = taw_stablity
            self._int += error
            new_taw = self._Kp * error + self._Ki * self._int + self._Kd * (error - self._diff)
            self._diff = error

        self._last_obs, r, d, info = self._env.step([new_taw])

        print("alpha : %f" % alpha)
        print("alpha_dot : %f" % alpha_dot)
        print("alpha_dotdot : %f" % alpha_dotdot)
        print("taw : %f" % new_taw)
        print("------------------------------------------")
        info.update({"action": new_taw})

        return self._last_obs, r, d, info