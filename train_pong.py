'''
!usr/bin/env python

Training a DDQN on pong
'''

import numpy as np
import torch


import gym
import dqn
import utils


env = gym.make('Pong-v0')
agent = dqn.DDQN(env, hyperparams = {'nframes' : 4, 'batch_size':10})
metrics, frames = utils.train_agent(agent, env, episodes = 10)

torch.save(agent.state_dict(), 'agent.state')