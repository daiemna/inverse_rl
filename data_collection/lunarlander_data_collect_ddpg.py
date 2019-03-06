from rllab.algos.ddpg import DDPG
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from inverse_rl.envs import register_custom_envs
from rllab.misc import logger, console
import os
import logging

# logger.print = logging.debug
# console.print = logging.debug 


def main(exp_name, ent_wt=1.0):
    register_custom_envs()
    env_name = 'LunarLanderContinuous-v3'
    env = GymEnv(env_name)
    policy = DeterministicMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))
    es = OUStrategy(env_spec=env.spec)
    qf = ContinuousMLPQFunction(env_spec=env.spec)

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=32,
        max_path_length=350,
        epoch_length=350,
        min_pool_size=350,
        n_epochs=600,
        discount=0.99,
        scale_reward=1.0/140.0,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
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
