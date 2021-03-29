import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

max_episode = 200
max_ep_step = 200
LR_A = 0.001
LR_C = 0.001
gamma = 0.9
memory_capacity = 10000
bitch_size=32
env_name='Pendulum-v0'


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.dims = dims

        self.data = np.zeros((self.capacity, self.dims))
        self.pointer = 0

    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, [action, reward], state_))
        indices = self.pointer % self.capacity
        self.data[indices, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, "Memory has not been fulfilled"
        index = np.random.choice(self.capacity, n)
        return self.data[index, :]


###############################  Actor  ####################################

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 30)
        self.fc1.weight.data.normal_(0., 0.3)
        self.out = nn.Linear(30, action_dim)
        self.out.weight.data.normal_(0., 0.3)

        self.action_bound = action_bound

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action = self.out(x)
        action = torch.tanh(action)

        scaled_action = action * self.action_bound

        return scaled_action


###############################  Critic  ####################################

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 30)
        self.fc1.weight.data.normal_(0., 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0., 0.1)

    def forward(self, x, u):
        x = self.fc1(torch.cat([x, u]))
        x = F.relu(x)
        q = self.out(x)

        return q


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 memory_capacity,
                 memory_dim,
                 learning_rate_A=0.001,
                 learning_rate_C=0.001,
                 reward_decay=0.9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.capacity = memory_capacity
        self.dim = memory_dim
        self.lr_A = learning_rate_A
        self.lr_C = learning_rate_C
        self.gamma = reward_decay

        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bound)
        self.critic = CriticNetwork(self.state_dim, self.action_dim)

        self.memory = Memory(self.capacity, self.dim)

        self.optimizer_A = torch.optim.Adam(self.actor.parameters(), self.lr_A)
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), self.lr_C)
