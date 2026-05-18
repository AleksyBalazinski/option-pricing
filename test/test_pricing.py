import unittest
from options import AmericanPut, AmericanCall, EuropeanCall, EuropeanPut
from bin_model import BinomialModel
import numpy as np


class TestAmerican(unittest.TestCase):
    def test_put(self):
        bin_model = BinomialModel(S0=50, N=2, dt=1, u=1.2, d=0.8, r=0.05)
        ame_put = AmericanPut(K=52)
        V0 = ame_put.price(bin_model)

        self.assertAlmostEqual(V0, 5.089, delta=1e-3)

    def test_call(self):
        bin_model = BinomialModel(S0=50, N=2, dt=1, u=1.2, d=0.8, r=0.05)
        ame_call = AmericanCall(K=42)
        V0 = ame_call.price(bin_model)

        self.assertAlmostEqual(V0, 13.248, delta=1e-3)


class TestEuropean(unittest.TestCase):
    def test_call(self):
        bin_model = BinomialModel(S0=20, N=2, dt=0.25, u=1.1, d=0.9, r=0.12)
        eur_call = EuropeanCall(K=21)
        V0 = eur_call.price(bin_model)

        self.assertAlmostEqual(V0, 1.282, delta=1e-3)

    def test_put(self):
        bin_model = BinomialModel(S0=20, N=2, dt=0.25, u=1.1, d=0.9, r=0.12)
        eur_put = EuropeanPut(K=21)
        V0 = eur_put.price(bin_model)

        self.assertAlmostEqual(V0, 1.059, delta=1e-3)

    def test_parity(self):
        S0 = 20
        K = 21
        r = 0.12
        N = 10
        dt = 0.25
        T = N * dt
        bin_model = BinomialModel(S0=S0, N=N, dt=dt, u=1.1, d=0.9, r=r)
        eur_call = EuropeanCall(K)
        eur_put = EuropeanPut(K)

        C = eur_call.price(bin_model)
        P = eur_put.price(bin_model)

        lhs = C + K * np.exp(-r * T)
        rhs = P + S0
        self.assertAlmostEqual(lhs, rhs, delta=1e-5)


class TestEuropeanHedging(unittest.TestCase):
    def test_hedging_call(self):
        bin_model = BinomialModel(S0=20, N=2, dt=0.25, u=1.1, d=0.9, r=0.12)
        eur_call = EuropeanCall(K=21)

        opt_tree, delta_tree, alpha_tree = eur_call.price_with_hedging(bin_model)
        #stock_tree = bin_model.generate_stock_tree()

        # wenzel A
        delta_A = delta_tree[0, 0]
        alpha_A = alpha_tree[0, 0]

        self.assertAlmostEqual(delta_A, 0.506425, delta=1e-4)
        self.assertAlmostEqual(alpha_A, -8.8462, delta=1e-2)

        # wezel B
        delta_B = delta_tree[0, 1]
        alpha_B = alpha_tree[0, 1]

        self.assertAlmostEqual(delta_B, 0.727273, delta=1e-3)
        self.assertAlmostEqual(alpha_B, -13.97441, delta=1e-5)

        # wezel C
        delta_C = delta_tree[1, 1]
        alpha_C = alpha_tree[1, 1]

        self.assertAlmostEqual(delta_C, 0.0, delta=1e-7)
        self.assertAlmostEqual(alpha_C, 0.0, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
