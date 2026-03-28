from collections import defaultdict

import numpy as np


class Agent:
    def __init__(self, env, gamma=0.9, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.cnts = defaultdict(lambda: 0)
        self.pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.V = defaultdict(lambda: 0.0)
        self.alpha = alpha
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

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
                data = (state, action, reward)
                done = self.env.get_is_done(next_state)
                self.memory.append(data)

                # 开始评估，计算target值
                target = reward + self.gamma * self.V[next_state]
                self.V[state] += (target - self.V[state]) * self.alpha
                if done:
                    print(f'Episode: {episode}, Reward: {reward}')
                    break
                state = next_state
