import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, n_actions, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc1.weight.data.normal_(0., 0.3)
        self.action_out = nn.Linear(20, n_actions)
        self.action_out.weight.data.normal_(0., 0.3)
        self.value_out = nn.Linear(20, 1)
        self.value_out.weight.data.normal_(0., 0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_out = self.action_out(x)
        value_out = self.value_out(x)
        activate_value = (action_out - action_out.mean()) + value_out

        return activate_value


class DuelingDQN(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=500,
                 memory_size=200,
                 batch_size=32,
                 e_greed_incerment=None,
                 dueling=True):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greed_incerment = e_greed_incerment
        self.dueling = dueling  # decided to dueling
        self.memory = np.zeros((self.memory_size, 2 * self.n_features + 2))

        self.learn_step_counter = 0

        self.epsilon = 0 if self.e_greed_incerment is None else self.epsilon_max

        self.eval_net, self.target_net = Net(self.n_actions, self.n_features), Net(self.n_actions, self.n_features)

        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)

        if np.random.uniform() < self.epsilon:
            action_state = self.eval_net.forward(observation)
            action = torch.max(action_state, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def store_transition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((state, [action, reward], state_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_parmeters_replace\n')
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[sample_index, :]

        b_state = torch.FloatTensor(batch_memory[:, :self.n_features])
        b_action = torch.LongTensor(batch_memory[:, self.n_features:self.n_features+1].astype(int))
        b_reward = torch.FloatTensor(batch_memory[:, self.n_features+1:self.n_features+2])
        b_state_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        q_eval = self.eval_net.forward(b_state).gather(1, b_action)
        q_next = self.eval_net.forward(b_state_).detach()
        q_target = self.gamma * q_next.max(1)[0] + b_reward
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
