from gym.envs.classic_control.acrobot import AcrobotEnv, rk4, wrap, bound
from gym import core, spaces
# from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
from rllab.misc import logger as logger

class AcrobotContinous(AcrobotEnv):
    discount_factor=0.95
    def __init__(self, max_episode_length=1000):
        AcrobotEnv.__init__(self)
        self._action_high=np.asarray([1.])
        self.action_space = spaces.Box(-self._action_high, self._action_high)
        self._max_episode_length = max_episode_length
        self._time_step = 0
        
    
    def _reset(self):
        self._time_step = 0
        return AcrobotEnv._reset(self)
    
    def _step(self, a):
        s = self.state
        torque = np.clip(a, -self._action_high, self._action_high)

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        self._time_step += 1
        terminal = True if terminal or self._time_step >= self._max_episode_length else False
        return (self._get_ob(), reward, terminal, {})
    
    def log_diagnostics(self, paths):
        # rew_dist = [traj.keys() for traj in paths]
        avg_disc_reward = [np.mean(traj['rewards'] * self.discount_factor ** (traj['rewards'].shape[0] - np.arange(traj['rewards'].shape[0]))) for traj in paths]
        avg_disc_reward = np.asarray(avg_disc_reward)
        # logger.log(str(rew_dist))
        logger.record_tabular('AvgDiscountedReturn', np.mean(avg_disc_reward))