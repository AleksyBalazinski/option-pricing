import numpy as np
from bin_model import BinomialModel
from recombinant_tree import RecombinantTree


class Option:
    def payoff(self, S):
        raise NotImplementedError()

    def price(self, model: BinomialModel):
        raise NotImplementedError()


class EuropeanOption(Option):
    def payoff(self, S):
        raise NotImplementedError()

    def calc_opt_tree(self, model: BinomialModel):
        q = 1 - model.p
        stock_tree = model.generate_stock_tree()
        N = model.N
        opt_tree = RecombinantTree(N)
        opt_tree.set_step(N, self.payoff(stock_tree.get_step(N)))

        for t in range(N - 1, -1, -1):
            for i in range(t + 1):
                expected = model.p * \
                    opt_tree[i, t + 1] + q * opt_tree[i + 1, t + 1]
                opt_tree[i, t] = np.exp(-1 * model.r * model.dt) * expected

        return opt_tree

    def price(self, model: BinomialModel):
        opt_tree = self.calc_opt_tree(model)

        return opt_tree[0, 0]

    def price_with_hedging(self, model: BinomialModel):
        N = model.N
        delta_tree = RecombinantTree(N - 1)
        alpha_tree = RecombinantTree(N - 1)

        opt_tree = self.calc_opt_tree(model)
        stock_tree = model.generate_stock_tree()
        for t in range(N - 1, -1, -1):
            for i in range(t + 1):
                V_up = opt_tree[i, t + 1]
                V_down = opt_tree[i + 1, t + 1]
                S_up = stock_tree[i, t + 1]
                S_down = stock_tree[i + 1, t + 1]

                delta = (V_up - V_down) / (S_up - S_down)
                alpha = opt_tree[i, t] - delta * stock_tree[i, t]

                alpha_tree[i, t] = alpha
                delta_tree[i, t] = delta

        return opt_tree, delta_tree, alpha_tree


class AmericanOption(Option):
    def payoff(self, S):
        raise NotImplementedError()

    def price(self, model: BinomialModel):
        q = 1 - model.p
        stock_tree = model.generate_stock_tree()
        N = model.N
        opt_tree = RecombinantTree(N)
        opt_tree.set_step(N, self.payoff(stock_tree.get_step(N)))

        for t in range(N - 1, -1, -1):
            for i in range(t + 1):
                expected = model.p * \
                    opt_tree[i, t + 1] + q * opt_tree[i + 1, t + 1]
                cont = np.exp(-1 * model.r * model.dt) * expected
                exercise = self.payoff(stock_tree[i, t])
                opt_tree[i, t] = max(cont, exercise)

        return opt_tree[0, 0]


class AmericanCall(AmericanOption):
    def __init__(self, K):
        self.K = K

    def payoff(self, S):
        return np.maximum(S - self.K, 0)


class AmericanPut(AmericanOption):
    def __init__(self, K):
        self.K = K

    def payoff(self, S):
        return np.maximum(self.K - S, 0)


class EuropeanCall(EuropeanOption):
    def __init__(self, K):
        self.K = K

    def payoff(self, S):
        return np.maximum(S - self.K, 0)


class EuropeanPut(EuropeanOption):
    def __init__(self, K):
        self.K = K

    def payoff(self, S):
        return np.maximum(self.K - S, 0)
