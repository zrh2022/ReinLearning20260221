from collections import defaultdict, deque

import numpy as np


# 改动：改成使用Q函数进行TD方法评估及改进

class Agent:
    def __init__(self, env, gamma=0.9, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.Q = defaultdict(lambda: 0.0)
        self.alpha = alpha
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.pi[state]
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
                data = (state, action, reward, done)
                self.memory.append(data)

                # 开始评估，计算target值
                if len(self.memory) > 1:
                    self.calculate_q_greedy()
                if done:
                    data = (next_state, None, None,  None)
                    self.memory.append(data)
                    self.calculate_q_greedy()
                    print(f'Episode: {episode}, Reward: {reward}')
                    break
                state = next_state

    def calculate_q_greedy(self):
        pre_state, pre_action, pre_reward, is_done = self.memory[0]
        state, action, _,  _ = self.memory[1]
        pre_key = (pre_state, pre_action)
        key = (state, action)

        if is_done:
            target = pre_reward
        else:
            target = pre_reward + self.gamma * self.Q[key]
        self.Q[pre_key] += (target - self.Q[pre_key]) * self.alpha

        # 贪婪化策略
        self.pi[pre_state] = self.greedy_policy(pre_state, 0.1, action_size=4)
