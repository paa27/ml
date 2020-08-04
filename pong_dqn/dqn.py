'''
Deep Q-Learning Network Agent
'''

import sys

from random import choices
from collections import deque
from copy import deepcopy

from IPython import display
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDQN():
    '''
    Double Deep Q-Learning Network
    
    
    Vocab:
    ------
    
    observation: immediate info provided by environment
    history: accumulation of state,action and observation
    state: F(history) to provide always same input length
    
    '''
    def __init__(self,env,hyperparams=None):
        
        if hyperparams is None:
            hyperparams = {}
        
        self.nactions = env.action_space.n
        
        self.epsilon = hyperparams.get('epsilon', 1) #Exploration probability
        self.epsilon_max = hyperparams.get('epsilon', 1)
        self.epsilon_min = hyperparams.get('epsilon_min',0.1)
        self.alpha = hyperparams.get('alpha',0.01) #Learning rate
        self.gamma = hyperparams.get('gamma',1) #Discount rate
        
        self.nframes = hyperparams.get('nframes', 4)
        
        self.replay_capacity = int(hyperparams.get('N',1e4))
        self.replay_memory = [] # Stores  {self.replay_capacity}x(state,action,reward,next_state)
        
        self.q_net = self._initialize_cnn()
        self.action_net = deepcopy(self.q_net)
        self.batch_size = hyperparams.get('batch_size',32)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.q_net.parameters(),
                                       lr = 0.001, weight_decay = 0.95, eps = 0.01)
        
        self.history = [] # Stores {self.nframes} most recent observations 
        self.steps = 0
        self.explore_steps = 50000
        self.copy_steps = 5000
        

    
    def _initialize_cnn(self):
        cnn = Net(channels_in = self.nframes, channels_out = self.nactions)
        return cnn
    
    def memorize(self, action, reward, observation, done):
        
        phi = _phi(self.history, nframes = self.nframes)
        self.history.append(observation)
        next_phi = _phi(self.history, nframes = self.nframes)      
        self.history = self.history[-10*self.nframes:] # 10 is simply an arbitrary history factor for debugging
        
        if (len(self.history) > self.nframes): 
            self.replay_memory.append([phi,
                                       action,
                                       reward,
                                       next_phi,
                                       done])
        
        self.replay_memory = self.replay_memory[-self.replay_capacity:]     
    
    def act(self):
        '''Epsilon-greedy policy
        '''
        
        if (len(self.history) >= self.nframes):
            state = _phi(self.history, nframes = self.nframes)

            if np.random.uniform(0,1)<self.epsilon:
                action = int(np.random.choice(self.nactions))
            else:
                with torch.no_grad():
                    action = self.action_net(torch.unsqueeze(state, 0)).argmax().item()
                    
        else:
            action = int(np.random.choice(self.nactions))
            
        self.steps += 1
        
        if self.steps <= self.explore_steps:
            self.epsilon += (self.epsilon_min - self.epsilon_max)/self.explore_steps
        if self.steps % self.copy_steps == 0:
            self._copy_net()
            
        return action
    
    def learn(self):
        
        if (len(self.replay_memory) >= self.batch_size):
            
            self.optimizer.zero_grad()
            samples = choices(self.replay_memory, k = self.batch_size)

            for sample in samples:
                state, action, reward, next_state, done = sample
                q_val = self.q_net(torch.unsqueeze(state, 0))
                with torch.no_grad():
                    target = self.q_net(torch.unsqueeze(state, 0))

                    if done:
                        target[0, action] = reward
                    else:
                        q_max = self.q_net(torch.unsqueeze(next_state, 0)).max().item()
                        target[0, action] = reward + self.gamma*q_max

                loss = self.criterion(q_val, target)
                loss.backward()

            self.optimizer.step()
            
    def _copy_net(self):
        self.action_net = deepcopy(self.q_net)
        
    def _eval_q(self, numpy_state):
        '''
        For debugging only
        '''
        tensor_state = torch.unsqueeze(torch.from_numpy(numpy_state).float(), 0)
        with torch.no_grad():
            result = self.q(tensor_state)
        return result.detach().numpy()
    
    def reset(self):
        pass
    
class Net(nn.Module):
    '''
    CNN used for handling pixel input
    '''
    
    def __init__(self, channels_in = 4, channels_out = 4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 8, 5)
        self.fc1 = nn.Linear(8 * 17 * 17, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, channels_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 17 * 17)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def _preprocess_frame(frame):
    '''
    Takes RGB image and returns cnn input.
    
    Params:
    -------
    frame: np.array, size: (210, 160, 3)
    
    Returns:
    --------
    np.array, size: (80,80)
    
    '''
    greyscale = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]) #(210,160,3) -> (210,160)
    #Hard coded for breakout
    downsampled = greyscale[::2,::2] #(210,160) -> (105,80)
    squared = downsampled[17:97:,] #(105,80) -> (80,80) , Includes the full 'play area'
    
    return squared

def _phi(history, nframes = 4):
        '''Extracts current state from learned history
        '''
        frames = history[-nframes:]
        state = torch.tensor([_preprocess_frame(frame) for frame in frames]).float()
        return state