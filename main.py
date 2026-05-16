import numpy as np


def get_down_factor(sigma, dt):
    return np.exp(-1 * sigma * np.sqrt(dt))


def get_up_factor(sigma, dt):
    return np.exp(sigma * np.sqrt(dt))


if __name__ == "__main__":
    print('Nothing to see here yet!')
