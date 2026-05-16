import unittest
from options import AmericanPut, AmericanCall, EuropeanCall, EuropeanPut
from bin_model import BinomialModel


class TestAmerican(unittest.TestCase):
    def test_put(self):
        bin_model = BinomialModel(S0=50, N=2, dt=1, u=1.2, d=0.8, r=0.05)
        amePut = AmericanPut(K=52)
        V0 = amePut.price(bin_model)

        self.assertAlmostEqual(V0, 5.089, delta=1e-3)

    def test_call(self):
        bin_model = BinomialModel(S0=50, N=2, dt=1, u=1.2, d=0.8, r=0.05)
        ameCall = AmericanCall(K=42)
        V0 = ameCall.price(bin_model)

        self.assertAlmostEqual(V0, 13.248, delta=1e-3)


class TestEuropean(unittest.TestCase):
    def test_call(self):
        bin_model = BinomialModel(S0=20, N=2, dt=0.25, u=1.1, d=0.9, r=0.12)
        eurCall = EuropeanCall(K=21)
        V0 = eurCall.price(bin_model)

        self.assertAlmostEqual(V0, 1.282, delta=1e-3)

    def test_put(self):
        bin_model = BinomialModel(S0=20, N=2, dt=0.25, u=1.1, d=0.9, r=0.12)
        eur_put = EuropeanPut(K=21)
        V0 = eur_put.price(bin_model)

        self.assertAlmostEqual(V0, 1.059, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
