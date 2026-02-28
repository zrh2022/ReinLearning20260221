import numpy as np


class Bandit:
    def __init__(self, arms):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
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