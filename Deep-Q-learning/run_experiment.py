from maze_env import Maze
from my_DQN import DeepQNet


def update():

    step = 0
    for episode in range(300):
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1

    print('game over')


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNet(env.n_actions,
                  env.n_features,
                  replace_target_iter=200,
                  memory_size=2000,
                  )
    env.after(100, update)
    env.mainloop()