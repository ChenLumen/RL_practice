import gym
from prioritized_replay_DQN import DQNPrioritizedReplay


env = gym.make('MountainCar-v0')
env = env.unwrapped
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
env.seed(21)
memory_size = 10000

RL = DQNPrioritizedReplay(n_actions=env.action_space.n,
                          n_features=env.observation_space.shape[0],
                          memory_size=memory_size,
                          e_greedy_increment=0.00005,)

total_steps = 0
for episode in range(20):
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        if done:
            reward = 10

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > memory_size:
            RL.learn()

        if done:
            print('episode', episode, 'finished')
            break

        observation = observation_
        total_steps += 1
