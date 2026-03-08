from collections import defaultdict
from Chapter2.part4_4.GridWorld import GridWorld
from Chapter2.part4_4.Agent import Agent

# 创建环境
env = GridWorld(4, 3)
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
pi_best = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
V = defaultdict(lambda: 0.0)
agent = Agent(env, gamma=0.9)


# 获取最优策略
pi = agent.policy_iter(env, pi, V)
print(pi)
# 可视化
env.render(V, pi)



