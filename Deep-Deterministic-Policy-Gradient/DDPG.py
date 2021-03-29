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
batch_size = 32
tau = 0.01  # soft replacement
env_name = 'Pendulum-v0'


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.dims = dims

        self.data = np.zeros((self.capacity, self.dims))
        self.pointer = 0

    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, action, [reward], state_))
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
        x = self.fc1(torch.cat([x, u], 1))
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
                 learning_rate_A=0.001,
                 learning_rate_C=0.001,
                 reward_decay=0.9,
                 batch_size=32,
                 tau=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.capacity = memory_capacity
        self.lr_A = learning_rate_A
        self.lr_C = learning_rate_C
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.tau = tau

        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bound)
        self.actor_target = ActorNetwork(self.state_dim, self.action_dim, self.action_bound)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(self.state_dim, self.action_dim)
        self.critic_target = CriticNetwork(self.state_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.memory = Memory(self.capacity, self.state_dim * 2 + self.action_dim + 1)

        self.optimizer_A = torch.optim.Adam(self.actor.parameters(), self.lr_A)
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), self.lr_C)

        self.loss_func = nn.MSELoss()

        self.num_actor_update_iteration = 0
        self.num_critic_update_iteration = 0
        self.num_training = 0

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        action = self.actor.forward(observation).data.numpy()
        action = action[0]

        return action

    def learn(self):
        # Sample from memory capacity
        batch_memory = self.memory.sample(self.batch_size)

        batch_state = torch.FloatTensor(batch_memory[:, :self.state_dim])
        batch_action = torch.FloatTensor(batch_memory[:, self.state_dim:self.state_dim + self.action_dim])
        batch_reward = torch.FloatTensor(batch_memory[:, -self.state_dim-1:-self.state_dim])
        batch_state_ = torch.FloatTensor(batch_memory[:, -self.state_dim:])

        # Compute the target Q value
        action_ = self.actor_target.forward(batch_state_)
        target_q = self.critic_target.forward(batch_state_, action_)
        target_q = batch_reward + (self.gamma * target_q).detach()

        # Get current Q estimate
        current_q = self.critic.forward(batch_state, batch_action)

        # Compute critic loss
        critic_loss = self.loss_func(current_q, target_q)

        self.optimizer_C.zero_grad()
        critic_loss.backward()
        self.optimizer_C.step()

        # Compute actor loss
        actor_loss = -self.critic(batch_state, self.actor(batch_state)).mean()

        self.optimizer_A.zero_grad()
        actor_loss.backward()
        self.optimizer_A.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1


###############################  training  ####################################

env = gym.make(env_name)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

ddpg = DDPG(state_dim=state_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            memory_capacity=memory_capacity,
            learning_rate_A=LR_A,
            learning_rate_C=LR_C,
            reward_decay=gamma,
            batch_size=batch_size,
            tau=tau)

var = 3  # Control exploration
t1 = time.time()

for episode in range(max_episode):
    observation = env.reset()
    ep_reward = 0
    for step in range(max_ep_step):
        env.render()

        action = ddpg.choose_action(observation)
        action = np.clip(np.random.normal(action, var), -2, 2)  # add randomness to action selection for exploration

        observation_, reward, done, info = env.step(action)

        reward /= 10

        ddpg.memory.store_transition(observation, action, reward, observation_)

        if ddpg.memory.pointer > memory_capacity:
            var *= 0.9995  # decay the action randomness
            ddpg.learn()

        observation = observation_
        ep_reward += reward

        if step == max_ep_step - 1:
            print('episode: ', episode, '  reward: ', int(ep_reward), '  explore: ', var)
            break

print('Running time: ', time.time() - t1)
