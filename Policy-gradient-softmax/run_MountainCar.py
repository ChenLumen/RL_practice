import gym
from policy_gradient import PolicyGradient


env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.02,
                    reward_decay=0.95,)

for episode in range(1000):
    observation =  env.reset()
    env.render()

    action = RL.choose_action(observation)

    observation_, reward, done, info = env.step(action)

    RL.store_transition(observation, action, reward)

    if done:
        ep_rs_sum = sum(RL.ep_rs)
        if 'running_conuter' not in globals():
            running_counter = ep_rs_sum
        else:
            running_counter = running_counter * 0.99 + ep_rs_sum * 0.01

        print('episode: ', episode, '  reward: ', int(running_counter))

        RL.learn()

        break

    observation = observation_
