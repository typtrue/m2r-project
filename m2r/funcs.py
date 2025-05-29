import numpy as np
import scipy.special as sci
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def hankel_estimate_smallx(alpha, x):
    G = np.euler_gamma
    PI = np.pi
    if alpha == 0:
        return 1 + 2.0j * (G + np.log(x/2)) / PI
    elif alpha == 1:
        return 2.0j / (PI * x) + 1.0j * x * (G - 0.5 + np.log(x/2)) / PI + x/2
    raise NotImplementedError(f"alpha must be 0 or 1, got {alpha}")


def hankel_estimate_largex(alpha, x):
    if alpha == 1:
        raise NotImplementedError(f"alpha must be 0, got {alpha}")
    PI = np.pi
    return np.sqrt(2 / (PI*x)) * np.exp(1.0j * (x - PI / 4))


def hankel(alpha, x):
    return sci.hankel1(alpha, x)


def trapezium_rule(n, x0, xn, f):
    xs = np.linspace(x0, xn, n)
    samples = np.array([f(x) for x in xs])
    h = (xn - x0) / n
    slice = np.array(samples[1:-1]) * 2
    samples[1:-1] = slice
    return 0.5*h*sum(samples)


def G(r, r_0):
    return hankel(0, 1 * np.linalg.norm(r - r_0)) / 4.0j


def func(r_0):
    return lambda x: 1.0j * 1 * G(np.array([x, 0]), r_0)


def testpoint(r_0, n=10):
    ikG = func(r_0)
    return trapezium_rule(n, -1, 1, ikG)