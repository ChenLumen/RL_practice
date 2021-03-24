import numpy as np
import pandas as pd


class Q_learning_table():
    def __init__(self, actions, learning_rate=0.9, reward_deacy=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_deacy
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() >= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            # Choose the best action
            # print(self.q_table)
            state_actions = self.q_table.loc[observation, :]
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)

        return action

    def learn(self, state, action, reward, _state):
        self.check_state_exist(_state)
        q_predict = self.q_table.loc[state, action]
        if _state != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[_state, :].max()
        else:
            q_target = reward

        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
