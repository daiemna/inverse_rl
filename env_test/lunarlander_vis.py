# from rllab.algos.trpo import TRPO
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from inverse_rl.envs import register_custom_envs
from rllab.misc import logger, console
# import os
import logging
# import joblib
# import time
import numpy as np
import datetime
import pandas as pd

logger.print = logging.debug
console.print = logging.debug 


def main(exp_name, ent_wt=1.0):
    register_custom_envs()
    env_name = 'LunarLanderContinuous-v3'
    # pickle_path = 'data/LunarLanderContinuous_v3_data_rllab_PPO/exp_1/itr_1499.pkl'
    # pickle_path = 'data/acrobat_data_rllab_trpo/exp_1/itr_800.pkl'
    # iter_data = joblib.load(pickle_path)
    env = GymEnv(env_name)
    state_cols = ["state_" + str(i) for i in range(env.observation_space.shape[0])]
    nstate_cols = ["next_state_" + str(i) for i in range(env.observation_space.shape[0])]
    action_cols = ["action_" + str(i) for i in range(env.action_space.shape[0])]
    reward_col = 'reward'
    save_path = "data/lunarlander_demo/"
    
    while True:
        o = env.reset()
        disc_r = 0
        r_sum = 0
        done = False
        i = 0
        cols = list(state_cols) + action_cols + nstate_cols + [reward_col]
        df = pd.DataFrame(columns=cols)
        # print("New episode!")
        while not done:
            env.render()
            # a, _ = iter_data['policy'].get_action(o)
            a = env.action_space.sample()
            no, r, done, _ = env.step(a)
            
            df = df.append(dict(
                [t for t in zip(state_cols, o)] + 
                [t for t in zip(action_cols, a)] +
                [t for t in zip(nstate_cols, no)] + 
                [(reward_col, r)]
            ), ignore_index=True)

            disc_r += r + disc_r * 0.9
            r_sum += r
            i += 1
            o = no
        # max_r = r
        print("iters : {} , disc_r : {} , sum_r : {}".format(i, disc_r, r_sum))
        if i >= 100:
            now_str = (
                    save_path +
                    'recording_{date:%Y-%m-%d_%H:%M:%S}.csv'
            ).format(date=datetime.datetime.now())
            df.to_csv(now_str)
            
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
