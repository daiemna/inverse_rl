import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
# import matplotlib.pylab as plt
import time
from pylogging import setup_logger, HandlerType
from logging import getLogger, getLevelName
from scripts.pid_pendulum import PendulumPID, do_experiment

logger = getLogger(__name__)
# param 0.6  0.9  0.6


def main():
    # setup_logger(log_directory='./logs', file_handler_type=HandlerType.ROTATING_FILE_HANDLER, allow_console_logging=True, console_log_level="DEBUG")
    min_val, max_val, stride = 0.0, 2.1, 0.1
    total_exp = int(((max_val-min_val)/stride)**3)
    param_grid = {
        "Kp": np.arange(min_val, max_val, stride),
        "Ki": np.arange(min_val, max_val, stride),
        "Kd": np.arange(min_val, max_val, stride)
    }

    target = 0
    pid_controller = PendulumPID(1, 0, 0, target=target)

    score_list = []
    i = 0
    print(param_grid)
    for params in list(ParameterGrid(param_grid)):
        avg_reward = do_experiment(pid_controller, params["Kp"], params["Ki"], params["Kd"])
        # print("Experiment")
        
        i += 1
        # pid_controller.env.terminate()
        logger.info("---------------------------Experiment {}/{}".format(i, total_exp))
        score_list.append((np.round(avg_reward, 2), np.round(params["Kp"], 2), np.round(params["Ki"], 2), np.round(params["Kd"], 2)))
    
    df = pd.DataFrame(data=score_list, columns=["score", "Kp", "Ki", "Kd"])
    max_score = max(df["score"])
    sel_df = df[df.score == max_score]

    logger.info("Max score is : \n{0}".format(max_score))
    logger.info("Winner is : \n {0}".format(sel_df))

    df.index = df["score"]
    df.sort_index(ascending=False, inplace=True)
    logger.info(df[["Kp", "Ki", "Kd"]].head(10))
    # df[["Kp", "Ki", "Kd"]].plot()
    # df.plot()
    # plt.show()


def main2():
    pid_cont = PendulumPID(-2.1, 0.0, 0.9, config_path="pid_constants.yml")
    done = False
    i = 0
    creward = 0.0
    while not done:
        pid_cont.env.render()
        obs, reward, done, info = pid_cont.step()
        creward += reward
        i += 1
        # time.sleep(0.4)
    logger.info("cumulative reward: {0}".format(creward))
    pid_cont.env.terminate()


if __name__ == "__main__":
    setup_logger(log_directory='./logs', file_handler_type=HandlerType.ROTATING_FILE_HANDLER, allow_console_logging=True, console_log_level=getLevelName("WARN"))
    main2()
