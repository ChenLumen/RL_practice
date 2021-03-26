import gym
from my_double_DQN import Double_DQN


env = gym.make('Pendulum-v0')
env = env.unwrapped

n_actions = 11
n_features = 3

RL = Double_DQN(n_actions=n_actions,
                n_features=n_features,
                learning_rate=0.005,
                e_greed_increment=0.001,
                memory_size=3000)

total_step = 0
observation = env.reset()
while True:
    env.render()
    action = RL.choose_action(observation)

    f_action = (action - (n_actions - 1) / 2) / ((n_actions - 1) / 4)  # convert to [-2 ~ 2] float actions
    observation_, reward, done, info = env.step([f_action])

    reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
    # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
    # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

    RL.store_transition(observation, action, reward, observation_)

    if total_step > 3000:
        RL.learn()

    if total_step - 3000 > 20000:
        break

    observation = observation_
    total_step += 1
