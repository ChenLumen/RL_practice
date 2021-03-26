import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_actions, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc1.weight.data.normal_(0, 0.3)
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        activate_value = self.out(x)
        return activate_value


class DeepQNet(object):

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_deacy=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_deacy
        self.max_epsilon = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.max_epsilon

        self.learn_step_count = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.eval_net, self.target_net = Net(self.n_actions, self.n_features), Net(self.n_actions, self.n_features)

        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.forward(x)
            # torch.max()[1].data 只返回variable中的数据部分（去掉Variable containing:）
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def store_transition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((state, [action, reward], state_))  # 沿着水平方向将数组堆叠起来。

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_count % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        self.learn_step_count += 1

        sample_index = np.random.choice(self.memory_size, self.batch_size)  # choice(a, size=None, replace=True, p=None)
        b_memory = self.memory[sample_index, :]

        b_state = torch.FloatTensor(b_memory[:, :self.n_features])
        b_action = torch.LongTensor(b_memory[:, self.n_features:self.n_features+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_features+1:self.n_features+2])
        b_state_ = torch.FloatTensor(b_memory[:, -self.n_features:])

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next = self.eval_net(b_state_).detach()
        q_target = self.gamma * q_next.max(1)[0].view(self.batch_size, 1) + b_reward

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
