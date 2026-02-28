import numpy as np

from Chapter1.part1_4.Bandit import Bandit
from Chapter1.part1_4.Agent import Agent
import matplotlib.pyplot as plt

"""
强化学习，章节1.4
修改内容：平均胜率实现
"""

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

num = 10
epsilon = 0.1
steps = 1000
runs = 200
all_rates = []
# 记录每次运行中最佳臂的真实胜率，用于后续统计分析
best_rates = []


for k in range(runs):
    bandit = Bandit(num)
    agent = Agent(epsilon, num)
    rates = []
    total_reward = 0
    best_rates.append(max(bandit.rates))
    for i in range(steps):
        action = agent.getAction()
        reward = bandit.play(action)
        agent.updateQs(action, reward)

        total_reward += reward
        rates.append(total_reward / (i+1))

    all_rates.append(rates)

avg_rates = np.mean(all_rates, axis=0)
# 绘制胜率图像
plt.figure(figsize=(10, 6))
plt.plot(avg_rates)
plt.xlabel('步骤')
plt.ylabel('胜率')
plt.title(f'ε-贪婪策略胜率变化 (ε={epsilon})')
plt.grid(True)
plt.show()

# 打印最终结果
print(f"运行次数: {runs}")
print(f"每轮步数: {steps}")
print(f"最终平均胜率: {avg_rates[-1]:.3f}")
print(f"平均最佳臂真实胜率: {np.mean(best_rates):.3f}")


