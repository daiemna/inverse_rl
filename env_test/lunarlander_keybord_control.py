#!/usr/bin/env python
from __future__ import print_function
from inverse_rl.envs import register_custom_envs
import sys, gym, time
import numpy as np
from pyglet.window import key as ks
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#
keyboard = ks.KeyStateHandler()
register_custom_envs()
env = gym.make('LunarLanderContinuous-v3' if len(sys.argv)<2 else sys.argv[1])

Kp = 0.1
Kt = 0.2
gamma=0.9
save_path = "data/lunarlander_demo/"
ACTIONS = env.action_space
RESET_ACTION = np.asarray([0., 0.])
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.
# print("ACTION high low : {} , {}".format(env.action_space.high, env.action_space.low))
human_wants_restart = False
human_sets_pause = False
state_cols = ["state_" + str(i) for i in range(env.observation_space.shape[0])]
nstate_cols = ["next_state_" + str(i) for i in range(env.observation_space.shape[0])]
action_cols = ["action_" + str(i) for i in range(env.action_space.shape[0])]
reward_col = 'reward'

def check_key():
    global human_agent_action, human_wants_restart, human_sets_pause, keyboard
    a = human_agent_action
    kbs = keyboard
    if kbs[ks.ENTER] or kbs[ks.RETURN]: human_wants_restart = True
    if kbs[ks.SPACE] : human_sets_pause = not human_sets_pause
    if kbs[ks.A]:
        a[1] += (-1. - a[1]) * Kp
    elif kbs[ks.D]:
        a[1] += (1. - a[1]) * Kp
    elif kbs[ks.Q]:
        a[1] += (-1. - a[1]) * Kp
    elif kbs[ks.E]:
        a[1] += (1. - a[1]) * Kp
    else:
        a[1] = 0.0
    
    if kbs[ks.W]:
        a[0] += (1. - a[0]) * Kt
    elif kbs[ks.Q]:
        a[0] += (1. - a[0]) * Kt
    elif kbs[ks.E]:
        a[0] += (1. - a[0]) * Kt
    else:
        a[0] = 0.0
    
    if np.any(a < ACTIONS.low) or np.any(a > ACTIONS.high): return
    human_agent_action = a


def mouse_scroll(x, y, sx, sy):
    # print(x, y, sx, sy)
    global human_agent_action
    a = human_agent_action
    a[0] += sy * 0.1
    human_agent_action = np.clip(a, ACTIONS.low, ACTIONS.high)

env.render()
env.unwrapped.viewer.window.on_mouse_scroll = mouse_scroll
env.unwrapped.viewer.window.push_handlers(keyboard)

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()

    skip = 0
    total_reward = 0
    total_timesteps = 0
    cols = list(state_cols) + action_cols + nstate_cols + [reward_col]
    df = pd.DataFrame(columns=cols)
    while 1:
        check_key()
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        nobser, r, done, info = env.step(a)
        df = df.append(dict(
            [t for t in zip(state_cols, obser)] + 
            [t for t in zip(action_cols, a)] +
            [t for t in zip(nstate_cols, nobser)] + 
            [(reward_col, r)]
        ), ignore_index=True)
        # df.append(dict([t for t in zip(action_cols, a)]))
        # df.append(dict([t for t in zip(nstate_cols, nobser)])) 
        # df.append({reward_col: r})
        total_reward = (r + total_reward * gamma)
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.05)
        obser = nobser
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    # df = pd.DataFrame(traj)
    # print(df.head())
    # df[list(state_cols) + action_cols + nstate_cols ].plot()
    # plt.show()
    if total_reward > -10 and total_timesteps >= 100:
        now_str = (
                save_path +
                'recording_{date:%Y-%m-%d_%H:%M:%S}.csv'
        ).format(date=datetime.datetime.now())
        df.to_csv(now_str)
    
        

print("ACTIONS={}".format(ACTIONS))
# print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
# print("No keys pressed is taking action 0")
os.makedirs(save_path, exist_ok=True)
while 1:
    human_agent_action = RESET_ACTION
    window_still_open = rollout(env)
    if window_still_open==False: break
