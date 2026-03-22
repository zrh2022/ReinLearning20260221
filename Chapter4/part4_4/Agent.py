from collections import defaultdict

from numpy import argmax


class Agent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma

    def eval_one_step(self, env, pi, V):
        for state in env.getStates():
            if state == env.wall_state or state == env.goal_state:
                continue

            # 只是会用到历史V[next_state],但是不会使用V[next_state]去累计状态价值
            # 每次必须重置next_V
            next_V = 0
            for action in env.action_space:
                next_state = env.getNextState(state, action)
                reward = env.getReward(next_state)
                next_V += pi[state][action] * (reward + self.gamma * V[next_state])
            V[state] = next_V

    def policy_eval(self, env, pi, V, threshold=0.0001):
        while True:
            old_V = V.copy()
            self.eval_one_step(env, pi, V)
            delta = 0.0
            for state in env.getStates():
                delta = max(delta, abs(V[state] - old_V[state]))
            if delta < threshold:
                break

    def get_best_policy(self, env, pi_best, V):
        for state in env.getStates():
            if state == env.wall_state or state == env.goal_state:
                continue

            action_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
            # 使用action_values避免多个动作都是1
            action_values = {}
            for action in env.action_space:
                next_state = env.getNextState(state, action)
                reward = env.getReward(next_state)
                action_values[action] = reward + self.gamma * V[next_state]

            # 注意不要使用np.argmax，因为会把整个字典当做1个元素，永远返回0
            best_action = max(action_values, key=action_values.get)
            action_probs[best_action] = 1.0
            pi_best[state] = action_probs

    def policy_iter(self, env, pi, V):
        while True:
            self.policy_eval(env, pi, V, threshold=0.0001)
            pi_best = {}
            self.get_best_policy(env, pi_best, V)
            if pi == pi_best:
                break
            pi = pi_best
        return pi_best
