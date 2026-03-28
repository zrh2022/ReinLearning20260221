from collections import defaultdict

import numpy as np

from Chapter6.part6_4.GridWorld import GridWorld
from Chapter6.part6_4.Agent import Agent


# 评估V函数
# 创建环境
env = GridWorld(4, 3)
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
agent = Agent(env)
episodes = 100000
agent.run(episodes)
env.render_Q(agent.Q)



