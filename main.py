from options import CustomCallOption
from bin_model import BinomialModel
from math import exp

K = 100
S0 = 100
N = 3
dt = 0.25
u = 1.25
d = 0.8
r = 0.08

p = (exp(r * dt) - d) / (u - d)

t1 = p**3 * (u**3 * S0 - K)
t2 = 3 * p**2 * (1 - p) * (u**2 * d * S0 - K)
num = t1 + t2
denom = p**3 + 3 * p**2 * (1 - p)
A = num / denom
print(f'{A=}')


customCall = CustomCallOption(K=K, A=A)
model = BinomialModel(S0=S0, N=N, dt=dt, u=u, d=d, r=r)
V0 = customCall.price(model)

print(f'{V0=}')
