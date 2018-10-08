# from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
# from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np
import logging
from .utils import read_yaml_file

logger = logging.getLogger(__name__)


class PendulumPID(object):

    def __init__(self, Kp, Kd, Kp_swing, **kwargs):
        self._env = TfEnv(GymEnv('Pendulum-v0',
                                 record_video=kwargs.get("record_video", False),
                                 record_log=kwargs.get("record_log", False)))
        # self._env = PendulumEnv()
        config_path = kwargs.get("config_path", None)
        config = {}
        if config_path is not None:
            config = read_yaml_file(config_path)
        
        self._alpha_tol = kwargs.get("stability_region", config.get("stablity-active-region", 0.0))
        self._Kmag = kwargs.get("swing_up_torque", config.get("max-swing-up-torque", 1.0))
        self._dt = kwargs.get("time_delta", config.get("time-delta", 1.0))
        self._length = kwargs.get("length", config.get("length", 1.0))
        self._mass = kwargs.get("mass", config.get("mass", 1.0))
        self._g = kwargs.get("g_val", config.get("gravitation", 10.0))
        self._target = kwargs.get("target", config.get("target-angle", 0.0))
        
        self.reset(Kp, Kd, Kp_swing, Ki=kwargs.get('Ki', 0.0))
        print("PID init")
        
    def reset(self, Kp, Kd, Kp_swing, Ki=0.0):
        self._int, self._diff, self._Ki = 0.0, 0.0, Ki
        self._Kp, self._Kd, self._Kp_swing = Kp, Kd, Kp_swing
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
        MAX_PE = self._mass * self._g * Ip
        # INR = self._mass * alpha_dotdot * Ip**2
        # Kp_swing = 0.02

        if abs(alpha - self._target) > self._alpha_tol:  # swing up
            # TODO: the units do not agree find a solution.
            new_taw = self._Kp_swing * np.sign(alpha_dot) * (MAX_PE - PE)
            # new_taw = np.sign(alpha_dot) * (self._Kmag / (1 + np.exp(-self._Kp_swing * alpha)))
            error = alpha
        else:                                            # stabilization pid
            error = np.sign(alpha_dot) * (KE + PE)
            self._int += error * self._dt
            _Cd = (error - self._diff) / self._dt if self._dt > 0 else 0
            new_taw = self._Kp * error + self._Ki * self._int + self._Kd * _Cd
            self._diff = error
        self._last_obs, r, d, info = self._env.step([new_taw])
        info.update({"error": error, "action": [new_taw]})

        return self._last_obs, r, d, info


def do_experiment_theta(pid_controller, Kp, Ki, Kd, exp_count=10):
    avg_c_theta = 0.0
    for i in np.arange(exp_count):
        pid_controller.reset(Kp, Ki, Kd)
        done = False
        
        c_theta = 0.0
        it = 0
        while True:
            obs, reward, done, info = pid_controller.step()
            c_theta += np.arccos(obs[0])
            
            if it >= 400:
                break
            
            it += 1 
        avg_c_theta += c_theta
    return (avg_c_theta/float(exp_count))


def do_experiment_error(pid_controller, Kp, Kd, Kps, Ki=0.0, exp_count=10, max_iterations=200):
    avg_error = 0.0
    for i in np.arange(exp_count):
        pid_controller.reset(Kp, Kd, Kps, Ki=Ki)
        done = False
        
        c_error = 0.0
        it = 0
        while True:
            obs, reward, done, info = pid_controller.step()
            c_error += info["error"] ** 2
            
            if it >= max_iterations:
                break
            
            it += 1 
        avg_error += c_error
    return (avg_error/float(exp_count))


def do_experiment_reward(Kp=1.0, Kd=1.0, Kps=1.0, gamma=0.99, pid_controller=None, exp_count=10, n=6, iters=200):
    if pid_controller is None:
        pid_controller = PendulumPID(Kp, Kd, Kps)
    exp_data = np.zeros((exp_count,))
    for k in range(exp_count):
        avg_disc_reward = 0.0
        for j in np.arange(n):
            pid_controller._last_obs = pid_controller._env.reset()
            done = False
            
            d_reward = 0.0
            for i in range(iters):
                obs, reward, done, info = pid_controller.step()
                d_reward += reward * gamma ** (iters - i)
                
            avg_disc_reward += d_reward
        exp_data[k] = avg_disc_reward/float(n)
    return exp_data

def do_experiment_random(env, exp_count=10, n=6, iters=200):
    exp_data = np.zeros((exp_count,))
    for k in range(exp_count):
        avg_disc_reward = 0.0
        for j in np.arange(n):
            # pid_controller._last_obs = pid_controller._env.reset()
            env.reset()
            done = False
            d_reward = 0.0
            for i in range(iters):
                obs, reward, done, info = env.step()
                d_reward += reward * gamma ** (iters - i)
                
            avg_disc_reward += d_reward
        exp_data[k] = avg_disc_reward/float(n)
    return exp_data