import numpy as np
from bin_model import BinomialModel
from recombinant_tree import RecombinantTree


class Hedger:
    def __init__(self, u, d, p, disc):
        self.u = u
        self.d = d
        self.disc = disc
        self.p = p
        self.q = 1 - p

    def get_hedge_positions(self, t, opt_values, S):
        V_up = opt_values[:t + 1]
        V_down = opt_values[1:t + 2]

        S_up = S * self.u
        S_down = S * self.d

        delta = (V_up - V_down) / (S_up - S_down)

        continuation = self.disc * (
            self.p * V_up +
            self.q * V_down
        )

        alpha = continuation - delta * S

        return alpha, delta, continuation


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

        disc = np.exp(-1 * model.r * model.dt)
        for t in range(N - 1, -1, -1):
            for i in range(t + 1):
                expected = model.p * \
                    opt_tree[i, t + 1] + q * opt_tree[i + 1, t + 1]
                opt_tree[i, t] = disc * expected

        return opt_tree

    def price(self, model: BinomialModel):
        N = model.N
        p = model.p
        q = 1 - p
        disc = np.exp(-model.r * model.dt)

        ST = np.array([
            model.S0 * (model.u ** (N - i)) * (model.d ** i)
            for i in range(N + 1)
        ])

        opt_values = self.payoff(ST)

        for t in range(N - 1, -1, -1):
            opt_values = disc * (
                p * opt_values[:t + 1] +
                q * opt_values[1:t + 2]
            )

        return opt_values[0]

    def price_with_hedging(self, model: BinomialModel):
        N = model.N
        p = model.p
        q = 1 - p

        disc = np.exp(-model.r * model.dt)

        delta_tree = RecombinantTree(N - 1)
        alpha_tree = RecombinantTree(N - 1)

        S = np.array([
            model.S0 * model.u**(N - i) * model.d**i
            for i in range(N + 1)
        ])

        opt_values = self.payoff(S)
        hedger = Hedger(model.u, model.d, model.p, disc)

        for t in range(N - 1, -1, -1):
            S = disc * (p * S[:t + 1] + q * S[1:t + 2])
            alpha, delta, cont = hedger.get_hedge_positions(t, opt_values, S)

            delta_tree.set_step(t, delta)
            alpha_tree.set_step(t, alpha)

            opt_values = cont

        return delta_tree, alpha_tree


class AmericanOption(Option):
    def payoff(self, S):
        raise NotImplementedError()

    def analyze_exercise_nodes(self, model: BinomialModel):
        N = model.N
        p = model.p
        q = 1 - p
        disc = np.exp(-model.r * model.dt)

        exercise_tree = RecombinantTree(N, dtype=np.bool)

        stock_tree = model.generate_stock_tree()

        ST = stock_tree.get_step(N)
        final_payoff = self.payoff(ST)
        opt_values = final_payoff.copy()

        # Mark terminal nodes as True if they expire in-the-money
        exercise_tree.set_step(N, final_payoff > 0)

        for t in range(N - 1, -1, -1):
            S_t = stock_tree.get_step(t)
            exercise_val = self.payoff(S_t)

            continuation_val = disc * (
                p * opt_values[:t + 1] +
                q * opt_values[1:t + 2]
            )

            # A node is optimal to exercise if:
            # 1. Exercise value is >= continuation value
            # 2. It is actually in-the-money (exercise_val > 0)
            is_optimal_exercise = (
                exercise_val >= continuation_val) & (exercise_val > 0)

            exercise_tree.set_step(t, is_optimal_exercise)

            opt_values = np.maximum(exercise_val, continuation_val)

        return exercise_tree

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
                cont = np.exp(-1 * model.r * model.dt) * expected
                exercise = self.payoff(stock_tree[i, t])
                opt_tree[i, t] = max(cont, exercise)

        return opt_tree

    def price(self, model: BinomialModel):
        N = model.N
        p = model.p
        q = 1 - p

        disc = np.exp(-model.r * model.dt)

        S = np.array([
            model.S0 * (model.u ** (N - i)) * (model.d ** i)
            for i in range(N + 1)
        ])

        opt_values = self.payoff(S)

        for t in range(N - 1, -1, -1):
            S = disc * (p * S[:t + 1] + q * S[1:t + 2])

            continuation = disc * (
                p * opt_values[:t + 1] +
                q * opt_values[1:t + 2]
            )

            exercise = self.payoff(S)

            opt_values = np.maximum(exercise, continuation)

        return opt_values[0]

    def price_with_hedging(self, model: BinomialModel):
        N = model.N
        p = model.p
        q = 1 - p

        disc = np.exp(-model.r * model.dt)

        delta_tree = RecombinantTree(N - 1)
        alpha_tree = RecombinantTree(N - 1)

        S = np.array([
            model.S0 * model.u**(N - i) * model.d**i
            for i in range(N + 1)
        ])

        opt_values = self.payoff(S)
        hedger = Hedger(model.u, model.d, model.p, disc)

        for t in range(N - 1, -1, -1):
            S = disc * (p * S[:t + 1] + q * S[1:t + 2])
            alpha, delta, cont = hedger.get_hedge_positions(t, opt_values, S)

            delta_tree.set_step(t, delta)
            alpha_tree.set_step(t, alpha)

            opt_values = np.maximum(self.payoff(S), cont)

        return delta_tree, alpha_tree


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
