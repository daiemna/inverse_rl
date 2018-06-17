from scripts.pid_pendulum import PendulumPID
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
import matplotlib.pylab as plt
# import time


def main():
    min_val, max_val, stride = 0.0, 1.01, 0.01 
    param_grid = {
        "Kp": np.arange(min_val, max_val, stride),
        "Ki": np.arange(min_val, max_val, stride),
        "Kd": np.arange(min_val, max_val, stride)
    }
    target = 0
    score_list = []
    for params in list(ParameterGrid(param_grid)):
        pid_cont = PendulumPID(params["Kp"], params["Ki"], params["Kd"], target)
        done = False
        i = 0
        creward = 0.0
        while not done:
            obs, reward, done, info = pid_cont.step()
            creward += reward
            i += 1
        pid_cont.env.terminate()

        score_list.append((creward, params["Kp"], params["Ki"], params["Kd"]))
    
    df = pd.DataFrame(data=score_list, columns=["score", "Kp", "Ki", "Kd"])
    max_score = max(df["score"])
    sel_df = df[df.score == max_score]
    print("Max score is : {0}".format(max_score))
    print("Winner is : ")
    print(sel_df)
    # df.index = df["score"]
    # df[["Kp", "Ki", "Kd"]].plot()
    # df.plot()
    # plt.show()

if __name__ == "__main__":
    main()
