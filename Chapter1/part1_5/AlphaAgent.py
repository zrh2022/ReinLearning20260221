import numpy as np


class AlphaAgent:
    def __init__(self, epsilon=0.1, alpha=0.1, action_size=10):
        self.action_size = action_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.Qs = np.zeros(action_size)
        self.Ns = np.zeros(action_size)

    def updateQs(self, action, reward):
        self.Ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def getAction(self):
        rate = np.random.rand()
        if rate < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.Qs)

        return action
