import scipy.special as sci_sp
import scipy.integrate as sci_int
import numpy as np
from typing import Sequence
import sympy as sp
from funcs import trapezium_rule
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


class HelmholtzSystem:
    def __init__(self, wave_no: float, bcs="D") -> None:
        self.k = wave_no

        y = sp.symbols('y')
        self.w_inc = sp.exp(1.0j * wave_no * y)

        self.bcs = bcs

        if bcs not in "DN":
            raise NotImplementedError

        self.c, self.B = self.init_problem()

        self.coeffs = np.linalg.solve(self.B, self.c)

    def G(self, x, y):
        return sci_sp.hankel1(0, self.k * np.linalg.norm(x - y)) / 4.0j

    def DG(self, x, y):
        return self.k * sci_sp.hankel1(1, self.k * np.linalg.norm(x - y)) / 4.0j

    def D2G(self, x, y):
        return (self.k ** 2) * (sci_sp.hankel1(0, self.k * np.linalg.norm(x - y)) - sci_sp.hankel1(2, self.k * np.linalg.norm(x - y))) / 8.0j

    def init_problem(self):
        bc = self.bcs
        N = np.floor(10 * self.k)
        eps = 10**(-14)
        nodes = np.arange(-1 + 1/N, 1, 2/N)
        print(nodes)
        print(len(nodes))
        if bc == "D":
            c = -np.ones(N, dtype=np.complex64)  # boundary condition u_i(x) = 1 on gamma
        elif bc == "N":
            c = -1.0j * self.k * np.ones(N, dtype=np.complex64)
        A = np.zeros((N, N), dtype=np.complex64)
        for i in range(N):
            def dG_r(x): return self.DG(nodes[i], x).real
            def dG_i(x): return self.DG(nodes[i], x).imag
            for j in range(N):
                if i != j:
                    A[i, j] = sci_int.quad(dG_r, nodes[j] - 1/N, nodes[j] + 1/N)[0] + 1.0j*sci_int.quad(dG_i, nodes[j] - 1/N, nodes[j] + 1/N)[0]
                else:
                    A[i, j] = sci_int.quad(dG_r, nodes[j] - 1/N + eps, nodes[j] + 1/N + eps)[0] + 1.0j*sci_int.quad(dG_i, nodes[j] - 1/N + eps, nodes[j] + 1/N + eps)[0]
            print(A[i,i])
        if bc == "D":
            B = A + np.identity(N) / 2
        elif bc == "N":
            B = A - np.identity(N) / 2
        return c, B  # type: ignore

    def weight(self, x):
        N = np.ceil(10*self.k)
        y = int((1 + x) // (2 / N))
        if y == N:
            y = int(N - 1)
        return self.coeffs[y]

    def u_scat(self, r):  # easily optimisable (define piecewise function)
        # N = np.ceil(10 * self.k)
        # nodes = np.linspace(-1 + 1/(2*N), 1 - 1/(2*N), N)

        bc = self.bcs

        if bc == "D":
            def f_r(x): return (self.DG(r, np.array([x, 0])) * self.weight(x)).real
            def f_i(x): return (self.DG(r, np.array([x, 0])) * self.weight(x)).imag
        elif bc == "N":
            def f_r(x): return (self.G(r, np.array([x, 0])) * self.weight(x)).real
            def f_i(x): return (self.G(r, np.array([x, 0])) * self.weight(x)).imag

        return sci_int.quad(f_r, -1, 1)[0] # + 1.0j * sci_int.quad(f_i, -1, 1)[0]

        # sum = 0

        # for i in range(N):
        #     def dG_phi(x): return self.DG(r, np.array([x, 0])) * self.coeffs[i]

        #     sum += trapezium_rule(100, nodes[i] - 1/(2*N), nodes[i] + 1/(2*N), dG_phi)

        # return sum

k = 10

sys = HelmholtzSystem(k)

print(sys.G(np.array([0, 1]), np.array([1, 1])))

print(sys.u_scat(np.array([0, 1])))

n = 100

xvals = np.linspace(-3, 3, n)
yvals = np.linspace(-3, 3, n)

x, y = np.meshgrid(xvals, yvals)


stack = np.column_stack((x.flatten(), y.flatten()))

z_vals = np.ones(len(stack))

for i in range(len(z_vals)):
    z_vals[i] = sys.u_scat(stack[i]).real  # + np.exp(1.0j * k * stack[i][1])
    if i % 100 == 0:
        print(f"{i}/{len(z_vals)}")

max = np.max(z_vals)

z = np.reshape(z_vals, (-1, n))


print(z.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)  # type: ignore

ax.set_zlim(-5, 5)  # type: ignore
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()