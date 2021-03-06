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
    max_r = 1

    itr_num = 0
    itr_inc = 30
    init_max_trials=6
    norm_max_trials=4

    max_iters= 400
    trail_num = 1

    while itr_num < 211:
        o = env.reset()
        if itr_num < 1:
            max_trials=init_max_trials
        else:
            max_trials=norm_max_trials

        disc_r=0
        r_sum=0
        done=False
        i=0
        # pickle_path = '../gpirl/notebooks_lunarlander/plots/gpirl_400_iter_post_trainig/itr_{}.pkl'.format(itr_num)
        pickle_path = 'data/LunarLanderContinuous_v3_data_rllab_PPO/exp_1/itr_{}.pkl'.format(itr_num)
        iter_data = joblib.load(pickle_path)
        while i < max_iters:
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
        if trail_num % max_trials == 0:
            if itr_num < 1:
                trail_num =  norm_max_trials
            itr_num += itr_inc
            print("**************** itr number : {} **********************".format(itr_num))
        trail_num += 1


if __name__ == "__main__":
    # for i in range(4):
    main("exp_{}".format(1), ent_wt=0.1)   
