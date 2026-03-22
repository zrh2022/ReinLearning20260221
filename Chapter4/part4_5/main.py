from collections import defaultdict
from Chapter4.part4_5.GridWorld import GridWorld
from Chapter4.part4_5.Agent import Agent

# 创建环境
env = GridWorld(4, 3)
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
pi_best = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
V = defaultdict(lambda: 0.0)
agent = Agent(env, gamma=0.9)


# 获取最优策略
V = agent.policy_eval(env, pi, V)
agent.get_best_policy(env, pi_best, V)
print(pi_best)
# 可视化
env.render(V, pi_best)



