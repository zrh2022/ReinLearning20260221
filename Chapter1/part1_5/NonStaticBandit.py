import numpy as np


class NonStaticBandit:
    def __init__(self, arms):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if np.random.rand() < rate:
            return 1
        else:
            return 0


# bandit = Bandit(10)
# print(bandit.rates)
#
# Q = 0
# for i in range(300):
#     reward = bandit.play(0)
#     # print(reward)
#     Q += (reward - Q) / (i+1)
#
# print(Q)