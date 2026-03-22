from collections import defaultdict

import numpy as np

from Chapter4.part4_extra_grid_world.GridWorld import GridWorld
from Chapter4.part4_extra_grid_world.Agent import Agent

np.random.seed(42)
# 创建环境
env = GridWorld(12, 7)
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



