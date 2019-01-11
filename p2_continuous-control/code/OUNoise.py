import numpy as np


class OUNoise:
    """ Ornstein-Uhlenbeck noise """

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def _get_update(self):
        return self.theta * (self.mu - self.state) + self.sigma * np.random.random(len(self.state))

    def sample(self):
        """ Update internal state and return it as a noise sample """
        self.state += self._get_update()
        return self.state
