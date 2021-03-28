import gym
from policy_gradient import PolicyGradient


env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.02,
                    reward_decay=0.99,)

for episode in range(3000):
    observation = env.reset()

    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            print('episode: ', episode, '  reward: ', int(running_reward))

            RL.learn()

            break
        observation = observation_
