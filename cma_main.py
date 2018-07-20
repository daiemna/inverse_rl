import numpy as np
# from sklearn.model_selection import ParameterGrid
import pandas as pd
import matplotlib.pylab as plt
import time
from pylogging import setup_logger, HandlerType
from logging import getLogger, getLevelName
from scripts.pid_pendulum import PendulumPID, do_experiment_error
import cma
import os
import os.path as osp
import datetime as dt

logger = getLogger(__name__)


def main():
    target = 0
    pid_controller = PendulumPID(1, 0, 0, target=target)

    def test_on_env(params):
        return do_experiment_error(pid_controller, 1.0, params[0], params[1])

    es = cma.CMAEvolutionStrategy([0.00, -0.07], 1)
    try:
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [test_on_env(x) for x in solutions])
            es.logger.add(modulo=2)  # write data to disc to be plotted
            es.disp()
    except KeyboardInterrupt as e:
        print(e)
    es.result_pretty()
    cma.plot()


MAX_LAST_REWARD = -0.002
MAX_RECORDS = 1000
SAVE_DIR = "pid_recordings_200_iter/"
COL_NAMES = "state1,state2,state3,action,next_state1,next_state2,next_state3,reward".split(",")
MAX_ITERATIONS = 400


def test_pid():
    pid_cont = PendulumPID(1.0, 2.6182, 2.3527, config_path="pid_constants.yml")
    while True:
        pid_cont.env.reset()
        done = False
        
        creward = 1.0
        for i in np.arange(0, MAX_ITERATIONS):
            pid_cont.env.render()
            obs, reward, done, info = pid_cont.step()
            creward += reward
            # time.sleep(0.3)
        logger.info("cumulative reward: {0}".format(creward))


def collect_pid_data():
    # TODO: action values are more than 2
    # sweet spot without Kmag: -19.376, 6.872, 0.202
    pid_cont = PendulumPID(-19.376, 6.872, 0.202, config_path="pid_constants.yml")
    record_count = 0
    
    if not osp.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    while record_count < MAX_RECORDS:
        pre_obs = pid_cont.env.reset()
        done = False
        episode = np.zeros((MAX_ITERATIONS, len(pre_obs) * 2 + 1 + 1))
        reward = 1.0
        for i in np.arange(0, MAX_ITERATIONS):
            # pid_cont.env.render()
            obs, reward, done, info = pid_cont.step()
            # logger.debug(type(pre_obs))
            # logger.debug(type(info["action"]))
            # logger.debug(type(reward))
            episode[i, :] = np.concatenate((pre_obs, info["action"], obs, [reward]))
            pre_obs = obs
            # creward += reward
            # time.sleep(0.3)
        # logger.info("cumulative reward: {0}".format(creward))
        if abs(np.mean(episode[200:, 5])) < 0.025:
            df = pd.DataFrame(data=episode, columns=COL_NAMES)
            logger.debug(df.head())
            df.to_csv(osp.join(SAVE_DIR, "recording_" + str(dt.datetime.now()).replace(' ', '_').split('.')[0])+".csv", index=False)
            record_count += 1
            # break

        # pid_cont.env.terminate()


def visualize_pid_data():
    if not osp.exists(SAVE_DIR):
        logger.error("csv dir not found!")
    only_csv_files = [osp.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if osp.isfile(osp.join(SAVE_DIR, f)) and (f.find(".csv") > 0)]
    for file_path in only_csv_files:
        df = pd.read_csv(file_path)
        df[["next_state2"]].plot()
        avg = df["next_state2"].tail(200).mean()
        logger.debug("Average cos(theta) for last 200 iter: {}".format(avg))
        plt.show()


if __name__ == "__main__":
    setup_logger(log_directory='./logs', file_handler_type=HandlerType.ROTATING_FILE_HANDLER, allow_console_logging=True, console_log_level="INFO")

    test_pid()
