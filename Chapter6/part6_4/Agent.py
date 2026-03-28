from collections import defaultdict, deque

import numpy as np


# 改动：异策略性的SARSA

class Agent:
    def __init__(self, env, gamma=0.9, alpha=0.1, action_size=4):
        self.env = env
        self.gamma = gamma
        # 目标策略
        self.pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        # 行为策略
        self.b = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.Q = defaultdict(lambda: 0.0)
        self.alpha = alpha
        self.memory = deque(maxlen=2)
        self.action_size = action_size

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def greedy_policy(self, state, epsilon, action_size=4):
        qs = [self.Q[(state, action)] for action in range(action_size)]
        max_action = np.argmax(qs)

        base_prob = epsilon / action_size
        action_pobs = {action: base_prob for action in range(action_size)}
        action_pobs[max_action] += (1 - epsilon)
        return action_pobs

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

        # 贪婪化策略
        self.pi[state] = self.greedy_policy(state, 0, action_size=4)
        self.b[state] = self.greedy_policy(state, 0.1, action_size=4)

