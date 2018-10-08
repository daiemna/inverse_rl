# from gym import core, spaces
from gym.envs.box2d.lunar_lander import LunarLander
import numpy as np


class LunarLanderMod(LunarLander):
    continuous = True

    def _step(self, action): 
        naction = np.clip(action, self.action_space.low, self.action_space.high)
        state, reward, done, info = LunarLander._step(self, naction)
        reward = reward if np.all(naction == action) else reward - 10
        return state, reward, done, info