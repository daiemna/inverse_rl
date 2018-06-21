import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
# import matplotlib.pylab as plt
import time
from pylogging import setup_logger, HandlerType

from scripts.pid_pendulum import PendulumPID, do_experiment
# found params.
# Kp    Ki    Kd
# 0.23  0.51  0.28


def main():
    setup_logger(log_directory='./logs', file_handler_type=HandlerType.ROTATING_FILE_HANDLER, allow_console_logging=False, console_log_level="WARNING")
    min_val, max_val, stride = -1.0, 1.0, 0.1
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
    for params in list(ParameterGrid(param_grid)):
        avg_reward = do_experiment(pid_controller, params["Kp"], params["Ki"], params["Kd"])
        
        i += 1
        # pid_controller.env.terminate()
        print("---------------------------Experiment {0}/{1}".format(i, total_exp))
        score_list.append((np.round(avg_reward, 2), np.round(params["Kp"], 2), np.round(params["Ki"], 2), np.round(params["Kd"], 2)))
    
    df = pd.DataFrame(data=score_list, columns=["score", "Kp", "Ki", "Kd"])
    max_score = max(df["score"])
    sel_df = df[df.score == max_score]
    print("Max score is : {0}".format(max_score))
    print("Winner is : ")
    print(sel_df)
    df.index = df["score"]
    df.sort_index(ascending=False, inplace=True)
    print(df[["Kp", "Ki", "Kd"]].head(10))
    # df[["Kp", "Ki", "Kd"]].plot()
    # df.plot()
    # plt.show()


def main2():

    setup_logger(log_directory='./logs', file_handler_type=HandlerType.ROTATING_FILE_HANDLER, allow_console_logging=False, console_log_level="DEBUG")
    pid_cont = PendulumPID(2.0, 0, 0, config_path="pid_constants.yml")
    done = False
    i = 0
    creward = 0.0
    while not done:
        pid_cont.env.render(mode='human')
        obs, reward, done, info = pid_cont.step()
        # print("theta : {0} , torueq : {1}".format(np.arccos(obs[0]), info["action"]))
        creward += reward
        i += 1
        time.sleep(0.5)
    print("cumulative reward: {0}".format(creward))
    pid_cont.env.terminate()


if __name__ == "__main__":
    main2()
