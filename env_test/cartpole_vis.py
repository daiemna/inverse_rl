from rllab.algos.trpo import TRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from inverse_rl.envs import register_custom_envs
from rllab.misc import logger, console
import os
import logging
import joblib
import time
import numpy as np

logger.print = logging.debug
console.print = logging.debug 


def main(exp_name, ent_wt=1.0):
    register_custom_envs()
    env_name = 'Cartpole-v2'
    pickle_path = 'data/cartpole_data_rllab_trpo/exp_1/itr_1000.pkl'
    # pickle_path = 'data/acrobat_data_rllab_trpo/exp_1/itr_800.pkl'
    iter_data = joblib.load(pickle_path)
    env = GymEnv(env_name)
    max_r = 1
    while True:
        o = env.reset()
        disc_r=0
        r_sum=0
        done=False
        i=0
        # print("New episode!")
        while not done:
            env.render()
            a, _ = iter_data['policy'].get_action(o)
            o, r, done, _ = env.step(a)
            s = [np.arccos(o[0]) , np.arccos(o[1])]
            # r = -np.cos(s[0]) - np.cos(s[1] + s[0])
            disc_r += r * 0.99 ** (500-i)
            r_sum += r
            i += 1
        # max_r = r
        print("disc_r : {} , sum_r : {}".format(disc_r,r_sum))
            
    # policy = GaussianMLPPolicy(env_spec=env, hidden_sizes=(64, 64))
    # algo = TRPO(
    #     env=env,
    #     policy=iter_data['policy'],
    #     n_itr=1500,
    #     batch_size=4000,
    #     max_path_length=1000,
    #     discount=0.99,
    #     store_paths=True,
    #     entropy_weight=ent_wt,
    #     baseline=iter_data['baseline']
    # )
    # data_path = 'data/acrobat_data_rllab_trpo/%s/'%exp_name
    # os.makedirs(data_path, exist_ok=True)
    # logger.set_snapshot_dir(data_path)
    # algo.train()
    # logger.set_snapshot_dir(None)


if __name__ == "__main__":
    # for i in range(4):
    main("exp_{}".format(1), ent_wt=0.1)   
