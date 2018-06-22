import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
# import matplotlib.pylab as plt
# import time
from pylogging import setup_logger, HandlerType
from logging import getLogger, getLevelName
from scripts.pid_pendulum import PendulumPID, do_experiment
import cma

logger = getLogger(__name__)


def main():
    target = 0
    pid_controller = PendulumPID(1, 0, 0, target=target)

    def test_on_env(params):
        return do_experiment(pid_controller, params[0], params[1], params[2])

    es = cma.CMAEvolutionStrategy(3*[0.0], 0.5)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [test_on_env(x) for x in solutions])
        es.logger.add(modulo=2)  # write data to disc to be plotted
        es.disp()
    es.result_pretty()
    cma.plot()


def main2():
    pid_cont = PendulumPID(2.1, 0.02, -0.48265, config_path="pid_constants.yml")
    done = False
    i = 0
    creward = 0.0
    while True:
        pid_cont.env.render()
        obs, reward, done, info = pid_cont.step()
        creward += reward
        if i >= 400:
            break
        i += 1
        # time.sleep(0.3)
    logger.info("cumulative reward: {0}".format(creward))
    pid_cont.env.terminate()

if __name__ == "__main__":
    # setup_logger(log_directory='./logs', file_handler_type=HandlerType.ROTATING_FILE_HANDLER, allow_console_logging=True, console_log_level=getLevelName("WARN"))
    main2()
