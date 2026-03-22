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

            max_value = float('-inf')
            for action in env.action_space:
                next_state = env.getNextState(state, action)
                reward = env.getReward(next_state)
                value = reward + self.gamma * V[next_state]
                if value > max_value:
                    max_value = value
                    max_action = action
                    pi_best[state] = {max_action: 1.0}
