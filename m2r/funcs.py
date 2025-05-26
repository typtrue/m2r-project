import numpy as np
import scipy.special as sci
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def hankel_estimate(alpha, x):
    G = np.euler_gamma
    PI = np.pi
    if alpha == 0:
        return 1 + 2.0j * (G + np.log(x/2)) / PI
    elif alpha == 1:
        return 2.0j / (PI * x) + 1.0j * x * (G - 0.5 + np.log(x/2)) / PI + x/2
    raise NotImplementedError(f"alpha must be 0 or 1, got {alpha}")


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


n = 50

xvals = np.linspace(-20, 20, n)
yvals = np.linspace(-20, 20, n)

x, y = np.meshgrid(xvals, yvals)

z_vals = np.array([testpoint(r).imag for r in np.column_stack((x.flatten(), y.flatten()))])

print(z_vals.shape)

z = np.reshape(z_vals, (-1, n))

print(z.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)  # type: ignore

ax.set_zlim(-0.5, 0.5)  # type: ignore
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
