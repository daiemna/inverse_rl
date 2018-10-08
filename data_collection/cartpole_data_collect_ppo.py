from rllab.algos.ppo import PPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from inverse_rl.envs import register_custom_envs
from rllab.misc import logger, console
import os
import logging

logger.print = logging.debug
console.print = logging.debug 


def main(exp_name, ent_wt=1.0):
    register_custom_envs()
    env_name = 'Cartpole-v3'
    env = GymEnv(env_name)
    policy = GaussianMLPPolicy(env_spec=env, hidden_sizes=(64, 64))
    baseline = GaussianMLPBaseline(env_spec=env)
    algo = PPO(
        env=env,
        policy=policy,
        n_itr=1500,
        batch_size=8000,
        max_path_length=500,
        discount=0.99,
        store_paths=True,
        entropy_weight=ent_wt,
        baseline=baseline
    )
    data_path = 'data/%s_data_rllab_%s/%s/'%(env_name.replace('-', '_'), 
                                             str(algo.__class__.__name__), 
                                             exp_name)
    os.makedirs(data_path, exist_ok=True)
    logger.set_snapshot_dir(data_path)
    algo.train()
    logger.set_snapshot_dir(None)


if __name__ == "__main__":
    # for i in range(4):
    main("exp_{}".format(1), ent_wt=1.0)   
