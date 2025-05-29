import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.special as sci_sp
import scipy.integrate as sci_int
import numpy as np


class HelmholtzSystem:
    def __init__(self, wave_no: float, bcs="D") -> None:
        self.k = wave_no

        self.bcs = bcs

        # TODO: Add impedence boundary conditions?

        if bcs not in "DN":  # check if boundary conditions are dirichlet or neumann:
            raise NotImplementedError
        

        # solve using BEM for phi, our weight function
        self.c, self.B = self.init_problem()

        self.coeffs = np.linalg.solve(self.B, self.c)

        print(self.coeffs)

    def G(self, x, y):
        """Green function for 2D Helmholtz."""
        return 1.0j * sci_sp.hankel1(0, self.k * np.linalg.norm(x - y)) / 4

    def DG(self, r_0, r):
        """Partial derivative of Green function wrt. y in second variable."""
        w, x = r_0
        y, z = r
        # return self.k * sci_sp.hankel1(1, self.k * np.linalg.norm(x - y)) / 4.0j
        return 1.0j * self.k * (z - x) * sci_sp.hankel1(1, self.k * np.linalg.norm(r_0 - r)) / (4 * np.linalg.norm(r_0 - r))

    def D2G(self, r_0, r):
        """Second partial derivative of Green function, once wrt y in first variable, once wrt y in second."""
        w, x = r_0
        y, z = r

        s1 = (2 * (w - y)**2 * sci_sp.hankel1(1, self.k * np.linalg.norm(r_0 - r))) / (np.linalg.norm(r_0 - r) ** 3)

        s2 = (self.k * (x - z)**2 * (sci_sp.hankel1(0, self.k * np.linalg.norm(r_0 - r)) - sci_sp.hankel1(2, self.k * np.linalg.norm(r_0 - r)))) / (np.linalg.norm(r_0 - r) ** 2)

        return - (self.k * 1.0j / 8) * (s1 + s2)

    def init_problem(self):
        """Initialise the problem."""
        bc = self.bcs
        N = np.floor(10 * self.k)
        # eps = 10**(-15)

        # pick nodes on boundary based on wave number
        nodes = np.arange(-1 + 1/N, 1, 2/N)

        # set up vector for matrix eqn. based on bcs
        if bc == "D":
            c = -np.ones(N, dtype=np.complex64)  # boundary condition u_i(x) = 1 on gamma
        elif bc == "N":
            c = -1.0j * self.k * np.ones(N, dtype=np.complex64)

        A = np.zeros((N, N), dtype=np.complex64)
        if bc == "D":
            # dirichlet: u = 0 on boundary
            for i in range(N):
                def f_r(x): return (1.0j * self.k * self.G(np.array([0, nodes[i]]), np.array([0, x])) - self.DG(np.array([0, nodes[i]]), np.array([0, x]))).real
                def f_i(x): return (1.0j * self.k * self.G(np.array([0, nodes[i]]), np.array([0, x])) - self.DG(np.array([0, nodes[i]]), np.array([0, x]))).imag
                for j in range(N):
                    if i != j:
                        A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0] + 1.0j*sci_int.quad(f_i, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0]
                    else:
                        # A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N + eps, nodes[j] + 1/N + eps, limit=np.floor(40*self.k))[0] + 1.0j*sci_int.quad(f_i, nodes[j] - 1/N + eps, nodes[j] + 1/N + eps, limit=np.floor(40*self.k))[0]
                        A[i, j] = 0 + 0j
                print(A[i, i])
            B = np.identity(N) / 2 - A
        elif bc == "N":
            # neumann: du/dy = 0 on boundary
            for i in range(N):
                def f_r(x): return (self.k ** 2 * self.G(np.array([0, nodes[i]]), np.array([0, x])) - 1.0j * self.k * self.DG(np.array([0, x]), np.array([0, nodes[i]])) + self.D2G(np.array([0, nodes[i]]), np.array([0, x]))).real
                def f_i(x): return (self.k ** 2 * self.G(np.array([0, nodes[i]]), np.array([0, x])) - 1.0j * self.k * self.DG(np.array([0, x]), np.array([0, nodes[i]])) + self.D2G(np.array([0, nodes[i]]), np.array([0, x]))).imag
                for j in range(N):
                    if i != j:
                        A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0] + 1.0j*sci_int.quad(f_i, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0]
                    else:
                        # A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N + eps, nodes[j] + 1/N + eps, limit=np.floor(40*self.k))[0] + 1.0j*sci_int.quad(f_i, nodes[j] - 1/N + eps, nodes[j] + 1/N + eps, limit=np.floor(40*self.k))[0]
                        A[i, j] = 0 + 0j
                print(A[i,i])
            B = 1.0j * self.k * np.identity(N) / 2 + A
        return c, B  # type: ignore

    def weight(self, x):
        # use piecewise function for faster integrating to calculate u_s (O(n^2) as opposed to O(n^3))
        N = np.floor(10*self.k)
        y = int((1 + x) // (2 / N))
        if y == N:
            y = int(N - 1)
        return self.coeffs[y]

    def u_scat(self, r):
        
        def f_r(x): return ((self.DG(r, np.array([x, 0])) - 1.0j * self.k * self.G(r, np.array([x, 0]))) * self.weight(x)).real
        def f_i(x): return ((self.DG(r, np.array([x, 0])) - 1.0j * self.k * self.G(r, np.array([x, 0]))) * self.weight(x)).imag

        return sci_int.quad(f_r, -1, 1, limit=100)[0] # + 1.0j * sci_int.quad(f_i, -1, 1)[0]




k = 10

sys = HelmholtzSystem(k, "N")

print(sys.G(np.array([0, 1]), np.array([1, 1])))

print(sys.u_scat(np.array([0, 1])))

n = 50

xvals = np.linspace(-3, 3, n)
yvals = np.linspace(-3, 3, n)

x, y = np.meshgrid(xvals, yvals)


stack = np.column_stack((x.flatten(), y.flatten()))

print(stack)

z_vals = np.ones(len(stack))

for i in range(len(z_vals)):
    z_vals[i] = sys.u_scat(stack[i])
    if i % 100 == 0:
        print(f"{i}/{len(z_vals)}")

max = np.max(z_vals)

# z_v = [z_vals[i] / max + np.exp(1.0j * k * stack[i][1]).real for i in range(len(z_vals))]

z = np.reshape(z_vals, (-1, n))


print(z.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)  # type: ignore

ax.set_zlim(-5, 5)  # type: ignore
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()