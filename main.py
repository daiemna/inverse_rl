import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
# import matplotlib.pylab as plt
import time

from scripts.pid_pendulum import PendulumPID
# found params.
# Kp    Ki    Kd
# 0.23  0.51  0.28


def main():
    min_val, max_val, stride = 0.0, 1.0, 0.01
    total_exp = ((max_val-min_val)/stride)**3
    param_grid = {
        "Kp": np.arange(min_val, max_val, stride),
        "Ki": np.arange(min_val, max_val, stride),
        "Kd": np.arange(min_val, max_val, stride)
    }
    target = 0
    score_list = []
    i = 0
    for params in list(ParameterGrid(param_grid)):
        pid_cont = PendulumPID(params["Kp"], params["Ki"], params["Kd"], target)
        done = False
        
        creward = 0.0
        while not done:
            obs, reward, done, info = pid_cont.step()
            creward += reward
        i += 1
        pid_cont.env.terminate()
        print("---------------------------Experiment {0}/{1}".format(i, total_exp))
        score_list.append((creward, params["Kp"], params["Ki"], params["Kd"]))
    
    df = pd.DataFrame(data=score_list, columns=["score", "Kp", "Ki", "Kd"])
    max_score = max(df["score"])
    sel_df = df[df.score == max_score]
    print("Max score is : {0}".format(max_score))
    print("Winner is : ")
    print(sel_df)
    df.index = df["score"]
    print(df.head(10))
    # df[["Kp", "Ki", "Kd"]].plot()
    # df.plot()
    # plt.show()


def main2():
    # last search:
    # https://www.myphysicslab.com/pendulum/pendulum-en.html
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    # google: pendulum mass length speed relation
    # pid_cont = PendulumPID(0.23,  0.51,  0.28, 0.0)
    pid_cont = PendulumPID(0.2,  0.0,  0.0, 0.0)
    done = False
    i = 0
    creward = 0.0
    while not done:
        pid_cont.env.render()
        obs, reward, done, info = pid_cont.step()
        print("theta : {0} , torueq : {1}".format(np.arccos(obs[0]), info["action"]))
        creward += reward
        i += 1
        time.sleep(1)
    print("cumulative reward: {0}".format(creward))
    pid_cont.env.terminate()


if __name__ == "__main__":
    main2()
