import numpy as np
from bin_model import BinomialModel
from tree_plotter import plot_tree


def get_down_factor(sigma, dt):
    return np.exp(-1 * sigma * np.sqrt(dt))


def get_up_factor(sigma, dt):
    return np.exp(sigma * np.sqrt(dt))


if __name__ == "__main__":
    bin_model = BinomialModel(S0=50, N=10, dt=1, u=1.2, d=0.8, r=0.05)
    plot_tree(bin_model.generate_stock_tree())
