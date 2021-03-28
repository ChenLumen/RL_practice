import gym
import numpy as np
from dueling_DQN import DuelingDQN


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

memory_size = 3000
action_space = 25
feature_space = 3

RL = DuelingDQN(n_actions=action_space,
                n_features=feature_space,
                memory_size=memory_size,
                e_greed_incerment=0.001)

total_step = 0
observation = env.reset()
while True:
    env.render()
    action = RL.choose_action(observation)
    f_action = (action - (action_space - 1) / 2) / ((action_space - 1) / 4)  # [-2 ~ 2] float actions

    observation_, reward, done, info = env.step(np.array([f_action]))

    reward /= 10

    RL.store_transition(observation, action, reward, observation_)

    if total_step > memory_size:
        RL.learn()

    if total_step - memory_size > 15000:
        break

    observation = observation_
    total_step += 1
