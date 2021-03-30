import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gym
import math
import os

os.environ['OMP_NUM_THREADS'] = '1'

# Hyper-parameters
update_global_iter = 5
gamma = 0.9
max_episode = 3000
max_episode_step = 200

env = gym.make('Pendulum-v0')
N_state = env.observation_space.shape[0]
N_action = env.action_space.shape[0]


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(optimizer, local_net, global_net, done, b_state, state_, b_action, b_reward, gamma):
    if done:
        v_s_ = 0  # terminal
    else:
        v_s_ = local_net.forward(v_wrap(state_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for reward in b_reward[::-1]:  # reverse buffer r
        v_s_ = reward + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
        v_wrap(np.vstack(b_state)),
        v_wrap(np.array(b_action), dtype=np.int64) if b_action[0].dtype == np.int64 else v_wrap(np.vstack(b_action)),
        v_wrap(np.array(buffer_v_target)[:, None])
    )

    # calculate local gradients and push local parameters to global
    optimizer.zero_grad()
    loss.backward()
    for local_parameter, global_parameter in zip(local_net.parameters(), global_net.parameters()):
        global_parameter._grad = local_parameter.grad
    optimizer.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1

    with global_ep_r.get_lock():
        if global_ep_r.value == 0:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value += global_ep_r.value * 0.99 + ep_r * 0.01

    res_queue.put(global_ep_r.value)
    print(
        name, 'Ep: ', global_ep.value,
        '| Ep_r: %f' % global_ep_r.value,
    )


class ShareAdam(torch.optim.Adam):
    """
    Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
    """
    def __init__(self,
                 parameters,
                 learning_rate=1e-3,
                 betas=(0.9, 0.99),
                 eps=1e-8,
                 weight_decay=0):
        super(ShareAdam, self).__init__(params=parameters,
                                        lr=learning_rate,
                                        betas=betas, eps=eps,
                                        weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 200)
        self.mu = nn.Linear(200, self.action_dim)
        self.sigma = nn.Linear(200, self.action_dim)
        self.c1 = nn.Linear(self.state_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.fc1, self.mu, self.sigma, self.c1, self.v])

    def forward(self, x):
        fc1 = self.fc1(x)
        fc1 = F.relu6(fc1)
        mu = self.mu(fc1)
        mu = 2 * torch.tanh(mu)
        sigma = self.sigma(fc1)
        sigma = F.softplus(sigma) + 0.001  # avoid 0
        x = self.c1(x)
        x = F.relu6(x)
        values = self.v(x)

        return mu, sigma, values

    def choose_action(self, observation):
        self.training = False
        mu, sigma, _ = self.forward(observation)
        m = torch.distributions.Normal(mu.view(1, ).data, sigma.view(1, ).data)

        return m.sample().numpy()

    def loss_func(self, state, action, v_t):
        self.train()
        mu, sigma, value = self.forward(state)
        td_error = v_t - value
        c_loss = td_error.pow(2)

        m = torch.distributions.Normal(mu, sigma)
        log_prob = m.log_prob(action)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td_error.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()

        return total_loss


class Worker(mp.Process):
    def __init__(self,
                 global_net,
                 optimizer,
                 global_ep,
                 global_ep_r,
                 res_queue,
                 name):
        super(Worker, self).__init__()

        self.global_net = global_net
        self.optimizer = optimizer
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        self.name = 'w%i' % name

        self.local_net = Net(N_state, N_action)
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        total_step = 1
        while self.global_ep.value < max_episode:
            observation = env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            ep_r = 0.

            for step in range(max_episode_step):
                if self.name == 'w0':
                    env.render()
                action = self.local_net.choose_action(v_wrap(observation[None, :]))
                observation_, reward, done, info = env.step(action.clip(-2, 2))

                if step == max_episode_step - 1:
                    done = True

                ep_r += reward

                buffer_action.append(action)
                buffer_state.append(observation)
                buffer_reward.append((reward+8.1)/8.1)

                if total_step % update_global_iter == 0 or done:
                    push_and_pull(self.optimizer, self.local_net, self.global_net, done,
                                  buffer_state, observation_, buffer_action, buffer_reward, gamma)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:
                        record(self.global_ep, self.global_ep_r, ep_r, self.res_queue, self.name)

                        break

                observation = observation_
                total_step += 1

        self.res_queue.put(None)


if __name__ == '__main__':
    global_net = Net(N_state, N_action)  # global network
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = ShareAdam(global_net.parameters(), learning_rate=1e-4, betas=(0.95, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(global_net, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [w.join() for w in workers]
