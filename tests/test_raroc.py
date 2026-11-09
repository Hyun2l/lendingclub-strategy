import numpy as np
from lc import raroc


def test_basel_k_hand_value():
    # PD 1%, LGD 45%: R = 0.1216, K = 0.45 x (N(-1.3323) - 0.01) ~ 0.0366
    assert abs(raroc.basel_retail_k(np.array([0.01]), np.array([0.45]))[0] - 0.0366) < 5e-4


def test_k_increases_with_pd_and_lgd():
    k = raroc.basel_retail_k(np.array([0.01, 0.05, 0.10]), np.array([0.5, 0.5, 0.5]))
    assert (np.diff(k) > 0).all()
    assert raroc.basel_retail_k(np.array([0.05]), np.array([0.6]))[0] > raroc.basel_retail_k(np.array([0.05]), np.array([0.3]))[0]


def test_annual_pd_round_trip():
    pa = raroc.annual_pd(np.array([0.271]), np.array([36]))[0]
    assert abs(1 - (1 - pa) ** 3 - 0.271) < 1e-9
