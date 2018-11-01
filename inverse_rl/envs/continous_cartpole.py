# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random balance point cart pole problem."""
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym import spaces
# from gym.utils import seeding
import numpy as np
# from numpy import sin, cos, pi
import math
from rllab.misc import logger
import logging

log = logging.getLogger(__name__)
# log.setLevel('DEBUG')
logger.print = log.debug


class ContinuosCartPoleEnv(CartPoleEnv):

    def __init__(self, max_episode_length=500, random_stable_position=False):
        CartPoleEnv.__init__(self)
        self.action_high = np.asarray([self.force_mag])
        self.action_space = spaces.Box(-self.action_high, self.action_high)
        self._max_episode_length = max_episode_length
        self._time_step = 0
        self._stable_x = None
        if random_stable_position:
            self._rand_pos_max = self.x_threshold - 0.4
            self._stable_x = np.random.uniform(-self._rand_pos_max, self._rand_pos_max)
            # log.info("obs high : {}".format(self.observation_space.high))
            oh = np.hstack((self.observation_space.high, np.asarray([self._rand_pos_max])))
            self.observation_space = spaces.Box(-oh, oh)
        log.debug("Action Space {}".format(self.action_space))
        log.debug("Observations Space {}".format(self.observation_space))
        # else:       

    def _step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        action = np.clip(action, -self.action_high, self.action_high)
        state = self.state
        if self._stable_x is not None:
            x, x_dot, theta, theta_dot, _ = state
        else:
            x, x_dot, theta, theta_dot = state
        # force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (action + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians \
            or self._time_step >= self._max_episode_length-1
        done = bool(done)
        # x_pos_reward = 0.0
        # if self._stable_x is not None:
        #     x_pos_reward = abs(x - self._stable_x)
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                log.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        if self._stable_x is not None:
            reward -= abs(x - self._stable_x)
            self.state += (np.asarray([self._stable_x]),)
        log.debug("state : {}".format(self.state))
        self._time_step += 1
        return np.asarray(self.state), reward, done, {}
    
    def _reset(self):
        self._time_step = 0
        if self._stable_x is not None:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
            self._stable_x = np.random.uniform(-self._rand_pos_max, self._rand_pos_max)
            self.state[4] = self._stable_x
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)