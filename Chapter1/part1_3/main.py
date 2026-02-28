from Chapter1.part1_3.Bandit import Bandit
from Chapter1.part1_3.Agent import Agent
import matplotlib.pyplot as plt

"""
强化学习，章节1.3
修改内容：随机的胜率实现
"""


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

num = 10
epsilon = 0.1
steps = 1000
total_reward = 0
rates = []

bandit = Bandit(num)
agent = Agent(epsilon, num)

for i in range(steps):
    action = agent.getAction()
    reward = bandit.play(action)
    agent.updateQs(action, reward)

    total_reward += reward
    rates.append(total_reward / (i+1))

# 绘制胜率图像
plt.figure(figsize=(10, 6))
plt.plot(rates)
plt.xlabel('步骤')
plt.ylabel('胜率')
plt.title(f'ε-贪婪策略胜率变化 (ε={epsilon})')
plt.grid(True)
plt.show()

# 打印最终结果
print(f"总步数: {steps}")
print(f"总奖励: {total_reward}")
print(f"最终胜率: {rates[-1]:.3f}")
print(f"最佳臂的真实胜率: {max(bandit.rates):.3f}")


