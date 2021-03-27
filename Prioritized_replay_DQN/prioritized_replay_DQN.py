import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class sumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0       -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_P(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree

    epislon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = sumTree(capacity)

    def store(self, transition):
        map_p = np.max(self.tree.tree[-self.tree.capacity:])
        if map_p == 0:
            map_p = self.abs_err_upper
        self.tree.add(map_p, transition)  # set the max p for new p

    def sample(self, n):
        b_indx, b_memory, ISWeights = \
            np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_P / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # Importance-sampling weight
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_P  # for later calculate ISWeight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_P
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_indx[i], b_memory[i, :] = idx, data

        return b_indx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epislon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNPrioritizedReplay(object):

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.005,
                 reward_decay=0.9,
                 e_greed=0.9,
                 replace_target_iter=500,
                 memory_size=10000,
                 batch_size=32,
                 e_greedy_increment=None,
                 prioritized=True, ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greed
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.prioritized = prioritized  # decide to use double q or not

        self.epsilon = 0 if self.e_greedy_increment is None else self.epsilon_max

        self.learn_step_counter = 0

        self.eval_net, self.target_net = Net(self.n_actions, self.n_features), Net(self.n_actions, self.n_features)

        self.memory = Memory(capacity=memory_size)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.lr)

        self.loss_func = nn.MSELoss()

    def choose_action(self, observation):
        x = torch.unsqueeze(torch.FloatTensor(observation), 0)
        if np.random.uniform() < self.epsilon:
            action_state = self.eval_net(x)
            action = torch.max(action_state, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def store_transition(self, state, action, reward, state_):

        transition = np.hstack((state, [action, reward], state_))
        self.memory.store(transition)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

        b_state = torch.FloatTensor(batch_memory[:, :self.n_features])
        b_action = torch.LongTensor(batch_memory[:, self.n_features:self.n_features+1].astype(int))
        b_reward = torch.FloatTensor(batch_memory[:, self.n_features+1:self.n_features+2])
        b_state_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next = self.eval_net(b_state_).detach()
        q_target = self.gamma * q_next.max(1)[0].view(self.batch_size, 1) + b_reward

        errors = torch.abs(q_eval - q_target).data.numpy()
        self.memory.batch_update(tree_idx, errors)

        loss = (self.loss_func(q_eval, q_target) * torch.FloatTensor(ISWeights)).mean()
        # loss = (torch.FloatTensor(ISWeights) * F.mse_loss(q_eval, q_target)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
