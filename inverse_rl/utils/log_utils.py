import os
import random
import joblib
import json
import contextlib
import os.path as osp

import rllab.misc.logger as rllablogger
import tensorflow as tf
import numpy as np
import pandas as pd

from inverse_rl.utils.hyperparametrized import extract_hyperparams

@contextlib.contextmanager
def rllab_logdir(algo=None, dirname=None):
    if dirname:
        rllablogger.set_snapshot_dir(dirname)
    dirname = rllablogger.get_snapshot_dir()
    rllablogger.add_tabular_output(os.path.join(dirname, 'progress.csv'))
    if algo:
        with open(os.path.join(dirname, 'params.json'), 'w') as f:
            params = extract_hyperparams(algo)
            json.dump(params, f)
    yield dirname
    rllablogger.remove_tabular_output(os.path.join(dirname, 'progress.csv'))


def get_expert_fnames(log_dir, n=5):
    print('Looking for paths')
    import re
    itr_reg = re.compile(r"itr_(?P<itr_count>[0-9]+)\.pkl")

    itr_files = []
    for i, filename in enumerate(os.listdir(log_dir)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('itr_count')
            itr_files.append((itr_count, filename))

    itr_files = sorted(itr_files, key=lambda x: int(x[0]), reverse=True)[:n]
    for itr_file_and_count in itr_files:
        fname = os.path.join(log_dir, itr_file_and_count[1])
        print('Loading %s' % fname)
        yield fname


def load_experts(fname, max_files=float('inf'), min_return=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if hasattr(fname, '__iter__'):
        paths = []
        for fname_ in fname:
            tf.reset_default_graph()
            with tf.Session(config=config):
                snapshot_dict = joblib.load(fname_)
            paths.extend(snapshot_dict['paths'])
    else:
        with tf.Session(config=config):
            snapshot_dict = joblib.load(fname)
        paths = snapshot_dict['paths']
    tf.reset_default_graph()

    trajs = []
    for path in paths:
        obses = path['observations']
        actions = path['actions']
        returns = path['returns']
        total_return = np.sum(returns)
        if (min_return is None) or (total_return >= min_return):
            traj = {'observations': obses, 'actions': actions}
            trajs.append(traj)
    random.shuffle(trajs)
    print('Loaded %d trajectories' % len(trajs))
    return trajs


def load_latest_experts(logdir, n=5, min_return=None):
    return load_experts(get_expert_fnames(logdir, n=n), min_return=min_return)


def load_latest_experts_multiple_runs(logdir, n=5):
    paths = []
    for i, dirname in enumerate(os.listdir(logdir)):
        dirname = os.path.join(logdir, dirname)
        if os.path.isdir(dirname):
            print('Loading experts from %s' % dirname)
            paths.extend(load_latest_experts(dirname, n=n))
    return paths


def load_pendlum_pid_experts_csv(data_dir, n=5):
    files = [f for f in os.listdir(data_dir) if osp.isfile(osp.join(data_dir, f)) and f.endswith('.csv')]
    files = files[:n]
    state_var_names = 'state1,state2,state3'.split(',')
    action_name = 'action'
    trajectories = []
    for file_name in files:
        path = osp.join(data_dir, file_name)
        # log.debug(osp.exists(path))
        df = pd.read_csv(path)
        
        actions = df[action_name]
        actions[actions > 2] = 2
        actions[actions < -2] = -2
        df[action_name] = actions
        data_dict = dict(observations=df[state_var_names].values, actions=df[[action_name]].values)
        trajectories.append(data_dict)
    return trajectories