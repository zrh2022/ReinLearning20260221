import numpy as np

from Chapter1.part1_4_episilon.Bandit import Bandit
from Chapter1.part1_4_episilon.Agent import Agent
import matplotlib.pyplot as plt

"""
强化学习，章节1.4
修改内容：比较不同epsilon值的效果
"""

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

num = 10
steps = 1000
runs = 200


def getRates(epsilon):
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
            rates.append(total_reward / (i + 1))

        all_rates.append(rates)

    avg_rates = np.mean(all_rates, axis=0)
    avg_best_rates = np.mean(best_rates, axis=0)
    return avg_rates, avg_best_rates


avg_rates_e1, best_rate_e1 = getRates(0.01)
avg_rates_e2, best_rate_e2 = getRates(0.1)
avg_rates_e3, best_rate_e3 = getRates(0.3)

# 绘制胜率图像
plt.figure(figsize=(10, 6))
x_axis = np.arange(len(avg_rates_e1))
plt.plot(x_axis, avg_rates_e1, label='ε=0.01')
plt.plot(x_axis, avg_rates_e2, label='ε=0.1')
plt.plot(x_axis, avg_rates_e3, label='ε=0.3')
plt.xlabel('步骤')
plt.ylabel('胜率')
plt.title('不同ε值的ε-贪婪策略胜率对比')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终结果
print(f"运行次数: {runs}")
print(f"每轮步数: {steps}")
print(f"ε=0.01 最终平均胜率: {avg_rates_e1[-1]:.3f}")
print(f"ε=0.1 最终平均胜率: {avg_rates_e2[-1]:.3f}")
print(f"ε=0.3 最终平均胜率: {avg_rates_e3[-1]:.3f}")
