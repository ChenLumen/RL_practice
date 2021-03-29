import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym


max_episode = 3000
max_ep_steps = 100
gamma = 0.9
lr_A = 0.001
lr_C = 0.01

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_f = env.observation_space.shape[0]
N_a = env.action_space.n


class Actor_network(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Actor_network, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc1.weight.data.normal_(0., 0.1)
        self.out = nn.Linear(20, n_actions)
        self.out.weight.data.normal_(0., 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        activate_value = self.out(x)
        activate_value = torch.softmax(activate_value, dim=-1)  # use softmax to convert to probability

        return activate_value


class Critic_network(nn.Module):
    def __init__(self, n_features):
        super(Critic_network, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc1.weight.data.normal_(0., 0.1)
        self.out = nn.Linear(20, 1)
        self.out.weight.data.normal_(0., 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        activate_value = self.out(x)

        return activate_value


class ActorCritic(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate_A=0.001,
                 learning_rate_C=0.01,
                 reward_decay=0.9):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_A = learning_rate_A
        self.lr_B = learning_rate_C
        self.gamma = reward_decay

        self.actor_net, self.critic_net = Actor_network(self.n_features, self.n_actions), Critic_network(self.n_features)

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr_A)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr_C)

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)

        prob_weights = self.actor_net.forward(observation)
        # select action w.r.t the actions prob
        c = Categorical(prob_weights)
        action = c.sample()
        action = action.data.numpy().astype(int)[0]

        return action

    def learn(self, state, action, reward, state_):
        state, state_ = torch.unsqueeze(torch.FloatTensor(state), 0), torch.unsqueeze(torch.FloatTensor(state_), 0)

        v = self.critic_net.forward(state)
        v_ = self.critic_net.forward(state_)

        td_error = (reward + self.gamma * v_).detach() - v
        loss_critic = torch.square(td_error)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        acts_prob = self.actor_net.forward(state)
        log_prob = torch.log(acts_prob[0, action])
        loss_actor = -torch.mean(log_prob * td_error.detach())
        # loss = exp_v

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()


model = ActorCritic(n_actions=N_a,
                    n_features=N_f,
                    learning_rate_A=lr_A,
                    learning_rate_C=lr_C,
                    reward_decay=gamma)

for episode in range(max_episode):
    observation = env.reset()
    step = 0
    track_r = []

    while True:
        env.render()

        action = model.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        if done:
            reward = -20

        track_r.append(reward)

        model.learn(observation, action, reward, observation_)

        observation = observation_
        step += 1

        if done:
            ep_re_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_re_sum
            else:
                running_reward = running_reward * 0.95 + ep_re_sum * 0.05

            print('episode: ', episode, '  reward: ', int(running_reward))

            break
