from collections import defaultdict

import numpy as np


class Agent:
    def __init__(self, env, gamma=0.9, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.cnts = defaultdict(lambda: 0)
        self.pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.Q = defaultdict(lambda: 0.0)
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

    def greedy_policy(self, Q, state, epsilon, action_size=4):
        qs = [Q[(state, action)] for action in range(action_size)]
        max_action = np.argmax(qs)

        base_prob = epsilon / action_size
        action_pobs = {action: base_prob for action in range(action_size)}
        action_pobs[max_action] += (1 - epsilon)
        return action_pobs

    def eval(self):
        G = 0
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            key = state, action
            self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) * self.alpha
            # 注意更新策略
            self.pi[state] = self.greedy_policy(self.Q, state, 0.1, action_size=4)

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
                if done:
                    print(f'Episode: {episode}, Reward: {reward}')
                    self.eval()
                    break
                state = next_state
