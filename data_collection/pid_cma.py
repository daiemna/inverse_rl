# !/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime as dt
import os
import os.path as osp
from argparse import ArgumentParser
from logging import getLogger

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pylogging import HandlerType, setup_logger

import cma
from data_collection.pid_pendulum import PendulumPID, do_experiment_error
from data_collection.utils import read_yaml_file

logger = getLogger(__name__)
COL_NAMES = "state1,state2,state3,action,next_state1,next_state2,next_state3,reward".split(",")

MAX_RECORDS = 1000
SAVE_DIR = "pid_recordings_200_iter/"
MAX_ITERATIONS = 400


def cma_tune(config):
    pid_controller = PendulumPID(1, 0, 0, Ki=0, config_path=config.get("pid-constants", None))

    def test_on_env(params):
        return do_experiment_error(pid_controller, params[0], params[1], params[2], max_iterations=MAX_ITERATIONS)

    es = cma.CMAEvolutionStrategy([-20.266, 7.00, 0.202], 1.0)
    try:
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [test_on_env(x) for x in solutions])
            es.logger.add(modulo=2)  # write data to disc to be plotted
            es.disp()
    except KeyboardInterrupt as e:
        print(e)
    # es.result_pretty()
    logger.debug("mean : {}".format(es.mean))
    logger.debug("var : {}".format(es.sigma))
    logger.debug("best : {}".format(es.best.x))
    cma.plot()
    return es.best.x


def test_pid(config):
    # best param -19.376, 6.872, 0.202,
    pid_cont = PendulumPID(config["best"][0], config["best"][1], config["best"][2], config_path=config.get("pid-constants", None))
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


def collect_pid_data(config):
    # TODO: action values are more than 2
    # sweet spot without Kmag: -19.376, 6.872, 0.202
    pid_cont = PendulumPID(config["best"][0], 
                           config["best"][1], 
                           config["best"][2], 
                           config_path=config.get("pid-constants", None))
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
        logger.info("average of last 50 iter: {0}".format(np.mean(episode[50:, 5])))
        if abs(np.mean(episode[50:, 5])) < 0.01:
            df = pd.DataFrame(data=episode, columns=COL_NAMES)
            logger.debug(df.head())
            df.to_csv(osp.join(SAVE_DIR, "recording_" + str(dt.datetime.now()).replace(' ', '_').split('.')[0])+".csv", index=False)
            record_count += 1
            # break

        # pid_cont.env.terminate()


def visualize_pid_data(config):
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
    parser = ArgumentParser(description='Main Process.')
    parser.add_argument('--config-path', metavar='config_path', type=str,
                        help='path to config file.')
    parser.add_argument('--test-only', metavar='test_only', type=bool, default=False,
                        help='test the best params, default False.')
    parser.add_argument('--debug', metavar='debug', type=bool, default=True,
                        help='log level debug, default False.')
    parser.add_argument('--log-path', metavar='log_path', type=str, default="./logs",
                        help='log file location.')
    parser.add_argument('--collect-data', metavar='collect_data', type=bool, default=False,
                        help='if true saves data of config.max-records tuned runs.')
    args = parser.parse_args()

    if args.debug:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    setup_logger(log_directory=args.log_path, file_handler_type=HandlerType.ROTATING_FILE_HANDLER, allow_console_logging=True, console_log_level="INFO")
    
    print("args : {}".format(args))
    config = read_yaml_file(args.config_path)
    MAX_RECORDS = config.get("max-records", MAX_RECORDS)
    SAVE_DIR = config.get("save-dir", SAVE_DIR)
    MAX_ITERATIONS = config.get("max-iterations", MAX_ITERATIONS)

    if not args.test_only:
        config["best"] = cma_tune(config)
    try:
        test_pid(config)
    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        pass
    if config.get("collect-data", args.collect_data):
        collect_pid_data(config)
