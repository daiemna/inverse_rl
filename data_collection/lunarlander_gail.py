import logging
import os
import os.path as osp
# import random
# import sys
# import time
# from shutil import rmtree
# import joblib
# import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats as ss
import pandas as pd
import tensorflow as tf
from pylogging import HandlerType, setup_logger
# from tqdm import tnrange, tqdm_notebook

# from gpirl.lunarlander_features import genrate_features
# from gpirl.gpirl import GPIRL
# from gpirl.utils import plot_experiment_data, plot_experiment_data_min_max, do_experiment_iter_data
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import GAIL
# from inverse_rl.utils.log_utils import load_pendlum_pid_experts_csv
from inverse_rl.envs import register_custom_envs
# from data_collection.pid_pendulum import do_experiment_reward, PendulumPID

# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.gym_env import GymEnv
from rllab.misc import console as rlc
from rllab.misc import logger as rllog
# from rllab.algos.ppo import PPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

log = logging.getLogger(__name__)
rllog.print = log.debug
rlc.print = log.debug
register_custom_envs()
setup_logger(log_directory='./logs',
             file_handler_type=HandlerType.ROTATING_FILE_HANDLER,
             allow_console_logging=True,
             console_log_level="DEBUG",
             change_log_level={
                 'tensorflow': 'error',
                 'matplotlib': 'error',
                 'GP': 'error',
                 'gpirl': 'info',
                 'gpirl.utils2': 'error',
                 __name__: 'info',
                 'gym': 'error'
             })

data_path = 'data/lunarlander_demo/'
env_name = 'LunarLanderContinuous-v3'
state_var_names = 'state_0,state_1,state_2,state_3,state_4,state_5,state_6,state_7'
nstate_var_names = state_var_names.replace('state', 'next_state')
action_names = 'action_0,action_1'
log.debug("Column names : {},{},{}".format(state_var_names, action_names, nstate_var_names))
trajectories = []
experts = []
plot_path = 'plots/'
gail_iter_path = plot_path + "gail_test_itrs/"
# gail_with_model_iter_path = plot_path + "gail_220_samples_with_model_itrs/"
# gpirl_iter_path = plot_path + "gpirl_220_samples_itrs/"
# gpirl_iter_path = plot_path + 'gpirl_kmeans_220_samples_itrs/'

# base_policy_path = '../../inverse_rl/data/LunarLanderContinuous_v3_data_rllab_PPO/exp_1/itr_0.pkl'
if not osp.exists(plot_path):
    os.makedirs(plot_path)
# sample_count = 100
idc = 10
# gpirl_iters=1000
testing = False
# test_gamma=0.9
file_count = 220
file_offset = 0
t_iter = 1
batch_size = 8000
ent_wt = 0.1
max_path_length = 500
step_size = 0.01

if testing:    
    file_count = 6
    file_offset = 0

files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f)) and f.endswith('.csv')]
files = files[file_offset: file_offset+file_count]

# log.info(files)
env = GymEnv(env_name, record_video=False, record_log=False)
for file_name in files:
    path = osp.join(data_path, file_name)
#     log.debug(osp.exists(path))
    df = pd.read_csv(path)
    data_dict = dict(observations=df[state_var_names.split(",")].values, actions=df[action_names.split(",")].values)
#     log.debug("{} loaded df size : {}".format(file_name, df.shape))
    if np.any(df.shape == [0, 0]):
        log.debug(file_name)
    else:
        log.debug(df.shape)
    trajectories.append(df)
    experts.append(data_dict)
log.info("trajs : {}".format(len(trajectories)))

tf.reset_default_graph()

# if not osp.exists(gail_iter_path):
#         os.makedirs(gail_iter_path)
# rllog.set_snapshot_dir(gail_iter_path)
with tf.Session():
    env = TfEnv(env)
    irl_model = GAIL(env_spec=env.spec, expert_trajs=experts)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(64, 64))
    # policy._mean_network = iter_data['policy']._mean_network
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=t_iter,
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=75,
        irl_model_wt=1.0,
        entropy_weight=0.0,  # GAIL should not use entropy unless for exploration
        zero_environment_reward=True,
        baseline=GaussianMLPBaseline(env_spec=env.spec)
    )
    algo.train()
    # rllog.set_snapshot_dir(None)