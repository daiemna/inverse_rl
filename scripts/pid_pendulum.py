# from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
# from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np
import logging
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


class PendulumPID(object):

    def __init__(self, Kp, Ki, Kd, **kwargs):
        self._env = TfEnv(GymEnv('Pendulum-v0',
                                 record_video=kwargs.get("record_video", False),
                                 record_log=kwargs.get("record_log", False)))
        # self._env = PendulumEnv()
        config_path = kwargs.get("config_path", None)
        config = {}
        if config_path is not None:
            config = read_yaml_file(config_path)
        
        self._alpha_tol = kwargs.get("stability_region", config.get("stablity-active-region", 0.5))
        self._dt = kwargs.get("time_delta", config.get("time-delta", 1.0))
        self._length = kwargs.get("length", config.get("length", 1.0))
        self._mass = kwargs.get("mass", config.get("mass", 1.0))
        self._g = kwargs.get("g_val", config.get("gravitation", 10.0))
        self._target = kwargs.get("target", config.get("target-angle", 0.0))

        self.reset(Kp, Ki, Kd)
        print("PID init")
        
    def reset(self, Kp, Ki, Kd):
        self._int, self._diff, self._Ki = 0.0, 0.0, 0.0
        self._Kp, self._Kp_swing, self._Kd = Kp, Ki, Kd
        self._last_obs = self._env.reset()
        self._alpha_dot_prev = self._last_obs[2]
    
    @property
    def env(self):
        return self._env

    def step(self):
        Ip = self._length / 2.0
        alpha, alpha_dot = np.arccos(self._last_obs[0]), self._last_obs[2]
        # alpha_dotdot = (alpha_dot - self._alpha_dot_prev)/self._dt
        self._alpha_dot_prev = alpha_dot

        PE = self._mass * self._g * Ip * np.sin(alpha)
        KE = self._mass * alpha_dot**2 * self._length**2 * 0.5
        # INR = self._mass * alpha_dotdot * Ip**2
        # Kp_swing = 0.02
        

        if abs(alpha - self._target) > self._alpha_tol:  # swing up
            new_taw = self._Kp_swing * np.sign(alpha_dot) * (KE + (2 - PE))
        else:                                            # stabilization pid
            error = -np.sign(alpha_dot) * (KE + PE)
            self._int += error
            new_taw = self._Kp * error + self._Ki * self._int + self._Kd * (error - self._diff)
            self._diff = error
        # logger.debug("alpha : %f" % alpha)
        # logger.debug("alpha_dot : %f" % alpha_dot)
        # logger.debug("alpha_dotdot : %f" % alpha_dotdot)
        # logger.debug("KE : %f" % (KE))
        # logger.debug("Inertia : %f" % (INR))
        # logger.debug("PE : %f" % (PE))
        # logger.debug("taw : %f" % new_taw)
        # logger.debug("------------------------------------------")
        self._last_obs, r, d, info = self._env.step([new_taw])

        info.update({"action": new_taw})

        return self._last_obs, r, d, info


def do_experiment(pid_controller, Kp, Ki, Kd, exp_count=10):
    avg_creward = 0.0
    for i in np.arange(exp_count):
        pid_controller.reset(Kp, Ki, Kd)
        done = False
        
        creward = 0.0
        it = 0
        while True:
            obs, reward, done, info = pid_controller.step()
            creward += reward
            
            if it >= 400:
                break
            
            it += 1 
        avg_creward += creward
    return (avg_creward/float(exp_count))


def read_yaml_file(path):
    """Reads a yaml file."""
    with open(path, 'r') as stream:
        yml = YAML(typ='safe')
        return yml.load(stream)