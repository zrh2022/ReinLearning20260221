from collections import defaultdict, deque

import numpy as np


# 改动：样本模型的Q学习

class Agent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, alpha=0.1, action_size=4):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = defaultdict(lambda: 0.0)
        self.alpha = alpha
        self.memory = deque(maxlen=2)
        self.action_size = action_size

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, action] for action in range(self.action_size)]
            return np.argmax(qs)

    def get_best_policy(self, env):
        pi = {}
        for state in env.getStates():
            qs = [self.Q[(state, action)] for action in range(self.action_size)]
            max_action = np.argmax(qs)
            pi[state] = max_action
        return pi

    def reset(self):
        self.memory.clear()

    def run(self, episodes):
        for episode in range(episodes):
            state = self.env.start_state
            self.reset()

            while True:
                action = self.get_action(state)
                next_state = self.env.getNextState(state, action)
                reward = self.env.getReward(next_state)
                done = self.env.get_is_done(next_state)
                self.calculate_q_greedy(state, action, reward, next_state, done)
                if done:
                    print(f'Episode: {episode}, Reward: {reward}')
                    break
                state = next_state

    def calculate_q_greedy(self, state, action, reward, next_state, is_done):
        qs_next = [self.Q[next_state, next_action] for next_action in range(self.action_size)]
        qs_next_max = max(qs_next)
        if is_done:
            target = reward
        else:
            target = reward + self.gamma * qs_next_max
        # 更新Q值
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

