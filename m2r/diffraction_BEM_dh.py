import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.special as sci_sp
import scipy.integrate as sci_int
import numpy as np


class HelmholtzSystem:
    def __init__(self, wave_no: float, bcs="D") -> None:
        self.k = wave_no
        self.N = np.floor(10 * wave_no)
        self.bcs = bcs

        # check if boundary conditions are dirichlet or neumann
        if bcs not in "DN":
            raise NotImplementedError

        # solve using BEM for phi, our weight function
        self.c, self.B = self.BEM()
        self.weights = np.linalg.solve(self.B, self.c)

    def G(self, x, y):
        """Green function for 2D Helmholtz."""
        return 1.0j * sci_sp.hankel1(0, self.k * np.linalg.norm(x - y)) / 4

    def DG(self, r_0, r):
        """Partial derivative of Green function wrt. y in second variable."""
        w, x = r_0
        y, z = r

        return 1.0j * self.k * (z - x) * sci_sp.hankel1(1, self.k * np.linalg.norm(r_0 - r)) / (4 * np.linalg.norm(r_0 - r))

    def D2G(self, r_0, r):
        """Second partial derivative of Green function, once wrt y in first variable, once wrt y in second."""
        w, x = r_0
        y, z = r

        s1 = (2 * (w - y)**2 * sci_sp.hankel1(1, self.k * np.linalg.norm(r_0 - r))) / (np.linalg.norm(r_0 - r) ** 3)

        s2 = (self.k * (x - z)**2 * (sci_sp.hankel1(0, self.k * np.linalg.norm(r_0 - r)) - sci_sp.hankel1(2, self.k * np.linalg.norm(r_0 - r)))) / (np.linalg.norm(r_0 - r) ** 2)

        return - (self.k * 1.0j / 8) * (s1 + s2)

    def BEM(self):
        """Initialise the BIE and solve using BEM."""
        bc = self.bcs
        N = self.N

        # pick nodes on boundary based on wave number, evenly spaced
        nodes = np.arange(-1 + 1/N, 1, 2/N)

        # set up vector for matrix eqn. based on bcs
        if bc == "D":
            c = -np.ones(N, dtype=np.complex64)  # boundary condition u_i(x) = 1 on gamma
        elif bc == "N":
            c = -1.0j * self.k * np.ones(N, dtype=np.complex64)  # boundary condition du_i/dy = ik on gamma

        A = np.zeros((N, N), dtype=np.complex64)

        if bc == "D":
            # dirichlet problem:
            # 
            # WIP

            for i in range(N):
                # we require real and imaginary parts of the function as scipy.integrate.quad() supports real-valued functions only
                def f_r(x): return (1.0j * self.k * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - self.DG(np.array([nodes[i], 0]), np.array([x, 0]))).real
                def f_i(x): return (1.0j * self.k * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - self.DG(np.array([nodes[i], 0]), np.array([x, 0]))).imag

                for j in range(N):
                    if i != j:
                        A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0] + 1.0j*sci_int.quad(f_i, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0]
                    else:
                        # TODO: find analytical solution for i = j to avoid integrating over singularity
                        A[i, j] = 0 + 0j
                print(A[i, i])
            B = np.identity(N) / 2 - A

        elif bc == "N":
            # neumann problem:
            #
            # WIP

            for i in range(N):
                def f_r(x): return (self.k ** 2 * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - 1.0j * self.k * self.DG(np.array([x, 0]), np.array([nodes[i], 0])) + self.D2G(np.array([nodes[i], 0]), np.array([x, 0]))).real
                def f_i(x): return (self.k ** 2 * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - 1.0j * self.k * self.DG(np.array([x, 0]), np.array([nodes[i], 0])) + self.D2G(np.array([nodes[i], 0]), np.array([x, 0]))).imag
                for j in range(N):
                    if i != j:
                        A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0] + 1.0j*sci_int.quad(f_i, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(40*self.k))[0]
                    else:
                        # TODO: find analytical solution for i = j to avoid integrating over singularity
                        A[i, j] = 0 + 0j
                print(A[i, i])
            B = 1.0j * self.k * np.identity(N) / 2 + A
        return c, B

    def weight(self, x):
        """Define calculated piecewise weight function."""
        N = self.N
        y = int((1 + x) // (2 / N))
        if y == N:
            y = int(N - 1)
        return self.weights[y]

    def u_scat(self, r):
        """Calculate scattered wave at point r."""

        # list of intersections between discretised plates
        antinodes = np.linspace(-1, 1, self.N+1)

        sum = 0

        # integrating over each plate individually and taking the sum is slower in theory
        # however integrating over the whole boundary yields inconsistencies due to the weight
        # being discontinuous (convergence of quadrature is slow)
        # this also yields a more accurate estimate

        for i in range(self.N):
            def f_r(x): return ((self.DG(r, np.array([x, 0])) - 1.0j * self.k * self.G(r, np.array([x, 0]))) * self.weights[i]).real

            sum += sci_int.quad(f_r, antinodes[i], antinodes[i+1], limit=100)[0]

        return sum


#############
## TESTING ##
#############


# wave number
k = 5

sys = HelmholtzSystem(k, "D")

# resolution of graph
n = 100

xvals = np.linspace(-3, 3, n)
yvals = np.linspace(-3, 3, n)

x, y = np.meshgrid(xvals, yvals)

stack = np.column_stack((x.flatten(), y.flatten()))

z_vals = np.ones(len(stack))

for i in range(len(z_vals)):
    z_vals[i] = sys.u_scat(stack[i])
    if i % 100 == 0:
        print(f"{i}/{len(z_vals)}")

# temporary measure to renormalise waves
max = np.max(z_vals)

z = np.reshape(z_vals, (-1, n))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(-5, 5)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()