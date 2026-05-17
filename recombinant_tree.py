import numpy as np


class RecombinantTree:
    def __init__(self, N):
        self.N = N
        self.length = self.get_start_idx(N + 1)
        self.data = np.zeros(self.length)

    @staticmethod
    def get_start_idx(t):
        return t * (t + 1) // 2

    @staticmethod
    def get_flat_idx(i, t):
        return (t * (t + 1)) // 2 + i

    def get_step(self, t):
        start = self.get_start_idx(t)
        end = self.get_start_idx(t + 1)
        return self.data[start:end].copy()

    def set_step(self, t, value):
        start = self.get_start_idx(t)
        end = self.get_start_idx(t + 1)
        self.data[start:end] = value

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Two indices required")

        i, t = key
        return self.data[self.get_flat_idx(i, t)]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Two indices required")

        i, t = key
        self.data[self.get_flat_idx(i, t)] = value
