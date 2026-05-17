import numpy as np


def get_down_factor(sigma, dt):
    return np.exp(-1 * sigma * np.sqrt(dt))


def get_up_factor(sigma, dt):
    return np.exp(sigma * np.sqrt(dt))


class BinomialModel:
    def __init__(self, S0, N, dt, u, d, r):
        self.S0 = S0
        self.N = N
        self.dt = dt
        self.r = r
        self.u = u
        self.d = d
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)

    def generate_stock_tree(self):
        tree = np.zeros(shape=(self.N + 1, self.N + 1))
        for t in range(self.N + 1):
            for i in range(t + 1):
                tree[i, t] = self.d**i * self.u**(t - i) * self.S0

        return tree
