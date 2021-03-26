import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, n_actions, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc1.weight.data.normal_(0., 0.3)
        self.out = nn.Linear(20, n_actions)
        self.out.weight.data.normal_(0., 0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        activate_value = self.out(x)
        return activate_value


class Double_DQN(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_deacy=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=3000,
                 batch_szie=32,
                 e_greed_increment=None,
                 double_q=True,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_deacy
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_szie
        self.e_greed_increment = e_greed_increment
        self.double_q = double_q
        self.epsilon = 0 if self.e_greed_increment is None else self.epsilon_max

        self.learn_step_count = 0

        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self.eval_net, self.target_net = Net(self.n_actions, self.n_features), Net(self.n_actions, self.n_features)

        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), self.lr)

        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:
            action_state = self.eval_net.forward(x)
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
        if self.learn_step_count % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        self.learn_step_count += 1

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]

        b_state = torch.FloatTensor(b_memory[:, :self.n_features])
        b_action = torch.LongTensor(b_memory[:, self.n_features:self.n_features+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_features+1:self.n_features+2])
        b_state_ = torch.FloatTensor(b_memory[:, -self.n_features:])

        q_eval = self.eval_net(b_state).gather(1, b_action)
        eval_action = self.target_net(b_state_).max(1)[1].view(self.batch_size, 1)
        q_next = self.eval_net(b_state_).detach()
        q_next = q_next.gather(1, eval_action)
        q_target = self.gamma * q_next + b_reward

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
