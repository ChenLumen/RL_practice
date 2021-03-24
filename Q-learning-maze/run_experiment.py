from Maze_env import Maze
from Q_learning import Q_learning_table


def update():
    for episode in range(100):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(str(observation))

            _observation, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(_observation))

            observation = _observation

            if done:
                break


if __name__ == '__main__':
    env = Maze()
    RL = Q_learning_table(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()