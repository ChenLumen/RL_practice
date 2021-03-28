import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, n_actions, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc1.weight.data.normal_(0., 0.3)
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal_(0., 0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        activate_value = self.out(x)
        activate_value = F.softmax(activate_value, dim=1)  # use softmax to convert to probability

        return activate_value


class PolicyGradient(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.net = Net(self.n_actions, self.n_features)

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        prob_weights = self.net.forward(observation)
        # select action w.r.t the actions prob
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.detach().numpy().ravel())

        return action

    def store_transition(self, state, action, reward):
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(reward)

    def discount_and_norm_reward(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        return discounted_ep_rs

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = torch.FloatTensor(self.discount_and_norm_reward())

        action_prob = self.net.forward(torch.FloatTensor(self.ep_obs))

        n_length = len(self.ep_as)
        #  convert to one-hot
        on_hot = torch.zeros(n_length, self.n_actions).scatter_(1, torch.LongTensor(self.ep_as).view(-1, 1), 1)
        neg_log_prob = torch.sum(-torch.log(action_prob) * on_hot, dim=1)
        loss = torch.mean(neg_log_prob * discounted_ep_rs_norm)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
