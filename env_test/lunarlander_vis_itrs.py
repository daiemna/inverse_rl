# from rllab.algos.trpo import TRPO
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from inverse_rl.envs import register_custom_envs
from rllab.misc import logger, console
# import os
import logging
import joblib
# import time
import numpy as np
import datetime
import pandas as pd

logger.print = logging.debug
console.print = logging.debug 


def main(exp_name, ent_wt=1.0):
    register_custom_envs()
    env_name = 'LunarLanderContinuous-v3'
    env = GymEnv(env_name)
    itr_num = 400
    episode_length = 400

    while True:
        o = env.reset()
        disc_r=0
        r_sum=0
        done=False
        i=0
        pickle_path = '../gpirl/notebooks_lunarlander/plots/gpirl_400_iter_post_trainig/itr_{}.pkl'.format(itr_num)
        # pickle_path = 'data/LunarLanderContinuous_v3_data_rllab_PPO/exp_1/itr_{}.pkl'.format(itr_num)
        iter_data = joblib.load(pickle_path)
        while i < episode_length:
            env.render()
            a, _ = iter_data['policy'].get_action(o)
            o, r, done, _ = env.step(a)
            # s = [np.arccos(o[0]), np.arccos(o[1])]
            # r = -np.cos(s[0]) - np.cos(s[1] + s[0])
            disc_r += r * 0.99 ** (500-i)
            r_sum += r
            i += 1
            if done:
                break
        # max_r = r
        print("disc_r : {} , sum_r : {}".format(disc_r, r_sum))
        print("last x : {}".format(o[0]))
        print("iterations : {}".format(i))
        print("-------------------------")


if __name__ == "__main__":
    # for i in range(4):
    main("exp_{}".format(1), ent_wt=0.1)   
