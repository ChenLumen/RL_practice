import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym


class Actor(nn.Module):
    def __init__(self, n_features, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.fc1 = nn.Linear(n_features, 30)
        self.fc1.weight.data.normal_(0., 0.1)
        self.mu = nn.Linear(30, 1)
        self.mu.weight.data.normal_(0., 0.1)
        self.sigma = nn.Linear(30, 1)
        self.sigma.weight.data.normal_(0., 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        mu = self.mu(x)
        mu = torch.tanh(mu)

        sigma = self.sigma(x)
        sigma = F.softplus(sigma)

        mu, sigma = torch.squeeze(mu * 2), torch.squeeze(sigma + 0.1)
        normal_dist = torch.distributions.Normal(mu, sigma)
        print(normal_dist.sample())

        action = torch.clamp(normal_dist.sample(), self.action_bound[0], self.action_bound[1])

        return action, normal_dist


class Critic(nn.Module):
    def __init__(self, n_features):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(n_features, 30)
        self.fc1.weight.data.normal_(0., 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0., 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        activate_value = self.out(x)

        return activate_value


class ActorCritic(object):
    def __init__(self,
                 n_features,
                 action_bound,
                 learning_rate_A=0.001,
                 learning_rate_C=0.01,
                 reward_decay=0.9):
        self.n_features = n_features
        self.action_bound = action_bound
        self.lr_A = learning_rate_A
        self.lr_C = learning_rate_C
        self.gamma = reward_decay

        self.actor_net, self.critic_net = Actor(self.n_features, self.action_bound), Critic(self.n_features)

        self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), self.lr_A)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), self.lr_C)

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        action, _ = self.actor_net.forward(observation)
        action = action.data.numpy()

        return action

    def learn(self, state, action, reward, state_):
        state, state_ = torch.unsqueeze(torch.FloatTensor(state), 0), torch.unsqueeze(torch.FloatTensor(state_), 0)

        v = self.critic_net.forward(state)
        v_ = self.critic_net.forward(state_)

        td_error = (reward + self.gamma * v_).detach() - v  # TD_error = (r+gamma*V_next) - V_eval
        loss_critic = torch.square(td_error)

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        _, normal_dist = self.actor_net.forward(state)
        log_prob = normal_dist.log_prob(torch.FloatTensor(action))  # loss without advantage
        exp_v = log_prob * td_error.detach()  # advantage (TD_error) guided loss
        # Add cross entropy cost to encourage exploration
        exp_v += 0.01 * normal_dist.entropy()

        loss_actor = -exp_v

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()


max_episode = 1000
max_ep_steps = 200
gamma = 0.9
LR_A = 0.001
LR_C = 0.01

env = gym.make('Pendulum-v0')
env.seed(1)
env = env.unwrapped

n_features = env.observation_space.shape[0]
action_bound = env.action_space.high[0]

model = ActorCritic(n_features=n_features,
                    action_bound=[-action_bound, action_bound],
                    learning_rate_A=LR_A,
                    learning_rate_C=LR_C,
                    reward_decay=gamma)

for episode in range(max_episode):
    observation = env.reset()
    step = 0
    ep_rs = []

    while True:
        env.render()
        action = model.choose_action(observation)

        observation_, reward, done, info = env.step(np.array([action]))
        reward /= 10

        model.learn(observation, action, reward, observation_)

        observation = observation_
        step += 1

        ep_rs.append(reward)

        if step > max_ep_steps:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1

            print('episode: ', episode, '  reward: ', int(running_reward))

            break
