import numpy as np

from Chapter1.part1_5.NonStaticBandit import NonStaticBandit
from Chapter1.part1_5.EpsilonAgent import EpsilonAgent
from Chapter1.part1_5.AlphaAgent import AlphaAgent
import matplotlib.pyplot as plt

"""
强化学习，章节1.4
修改内容：非稳态问题
"""

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置固定的随机种子，确保结果可重现
np.random.seed(2026)

num = 10
epsilon = 0.1
alpha = 0.1
steps = 1000
runs = 200
all_rates1 = []
all_rates2 = []
# 记录每次运行中最佳臂的真实胜率，用于后续统计分析
best_rates1 = []
best_rates2 = []

for k in range(runs):
    bandit1 = NonStaticBandit(num)
    bandit2 = NonStaticBandit(num)
    epsilonAgent = EpsilonAgent(epsilon, num)
    alphaAgent = AlphaAgent(epsilon, alpha, num)
    rates1 = []
    rates2 = []
    total_reward1 = 0
    total_reward2 = 0

    for i in range(steps):
        # 使用稳态环境下的策略
        action1 = alphaAgent.getAction()
        reward1 = bandit1.play(action1)
        alphaAgent.updateQs(action1, reward1)

        total_reward1 += reward1
        rates1.append(total_reward1 / (i+1))

        # 使用非稳态环境下的策略
        action2 = epsilonAgent.getAction()
        reward2 = bandit2.play(action2)
        epsilonAgent.updateQs(action2, reward2)

        total_reward2 += reward2
        rates2.append(total_reward2 / (i+1))

    best_rates1.append(max(bandit1.rates))  # 添加这行
    best_rates2.append(max(bandit2.rates))  # 添加这行
    all_rates1.append(rates1)
    all_rates2.append(rates2)

avg_rates1 = np.mean(all_rates1, axis=0)
avg_rates2 = np.mean(all_rates2, axis=0)
# 绘制胜率图像
plt.figure(figsize=(10, 6))
x_axis = np.arange(len(avg_rates1))
plt.plot(x_axis, avg_rates1, label='ε策略')
plt.plot(x_axis, avg_rates2, label='α策略')
plt.xlabel('步骤')
plt.ylabel('胜率')
plt.title('不同策略对于非稳态环境的胜率对比')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终结果
print(f"ε策略最终平均胜率: {avg_rates1[-1]:.3f}")
print(f"α策略最终平均胜率: {avg_rates2[-1]:.3f}")
print(f"ε策略平均最佳臂胜率: {np.mean(best_rates1):.3f}")
print(f"α策略平均最佳臂胜率: {np.mean(best_rates2):.3f}")


