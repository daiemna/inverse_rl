from rllab.algos.ppo import PPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from inverse_rl.envs import register_custom_envs
from rllab.misc import logger, console
import os
from gym.envs.box2d.lunar_lander import LunarLanderContinuous
import logging

logger.print = logging.debug
console.print = logging.debug 


def main(exp_name, ent_wt=1.0):
    register_custom_envs()
    env_name = 'Acrobot-v2'
    env = GymEnv(env_name)
    policy = GaussianMLPPolicy(env_spec=env, hidden_sizes=(64, 64))
    algo = PPO(
        env=env,
        policy=policy,
        n_itr=1500,
        batch_size=8000,
        max_path_length=1000,
        discount=0.95,
        store_paths=True,
        entropy_weight=ent_wt,
        baseline=LinearFeatureBaseline(env_spec=env)
    )
    data_path = 'data/acrobat_data_rllab_ppo/%s/'%exp_name
    os.makedirs(data_path, exist_ok=True)
    logger.set_snapshot_dir(data_path)
    algo.train()
    logger.set_snapshot_dir(None)


if __name__ == "__main__":
    # for i in range(4):
    main("exp_{}".format(1), ent_wt=1.0)   
