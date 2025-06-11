import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.special as sci_sp
import scipy.integrate as sci_int
import numpy as np
import mayavi.mlab as mlab


class HelmholtzSystem:
    def __init__(self, wave_no: float, bcs="D") -> None:
        self.k = wave_no
        self.N = int(np.floor(10 * wave_no))
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

    def DG(self, r, r_0):
        """Partial derivative of Green function wrt. y in second variable."""
        w, x = r
        y, z = r_0

        return 1.0j * self.k * (x - z) * sci_sp.hankel1(1, self.k * np.linalg.norm(r - r_0)) / (4 * np.linalg.norm(r - r_0))

    def D2G(self, r, r_0):
        """Second partial derivative of Green function, once wrt y in first variable, once wrt y in second."""
        w, x = r
        y, z = r_0

        # s1 = (2 * sci_sp.hankel1(1, self.k * np.linalg.norm(r - r_0))) / (np.linalg.norm(r - r_0) ** 3)

        # s2 = (self.k * (sci_sp.hankel1(0, self.k * np.linalg.norm(r - r_0)) - sci_sp.hankel1(2, self.k * np.linalg.norm(r - r_0)))) / (np.linalg.norm(r - r_0) ** 2)

        # return (1.0j * self.k * (x - z) * (w - y) / 8) * (s1 - s2)

        s = sci_sp.hankel1(0, self.k * np.linalg.norm(r - r_0)) - sci_sp.hankel1(2, self.k * np.linalg.norm(r - r_0))

        return (1.0j * self.k ** 2 * (x - z) ** 2 * s) / (8 * np.linalg.norm(r - r_0) ** 2)

    def BEM(self):
        """Initialise the BIE and solve using BEM."""
        bc = self.bcs
        N = self.N

        # pick nodes on boundary based on wave number, evenly spaced
        nodes = np.arange(-1 + 1/N, 1, 2/N)

        # set up vector for matrix eqn. based on bcs
        if bc == "D":
            c = -np.ones(N, dtype=complex)  # boundary condition u_i(x) = 1 on gamma
        elif bc == "N":
            c = -1.0j * self.k * np.ones(N, dtype=complex)  # boundary condition du_i/dy = ik on gamma

        A = np.zeros((N, N), dtype=complex)

        if bc == "D":
            # dirichlet problem:
            # 
            # WIP

            # analytical solution for the diagonal
            # diag = - self.k * (1 / (2 * N) + 2.0j * (np.log(self.k / (2 * N)) + np.euler_gamma - 1) / (np.pi * N))
            diag = - (np.pi + 2.0j * (np.log(1 / N) + np.log(self.k / 2) + np.euler_gamma - 1)) / (2 * np.pi * N)

            for i in range(N):
                # we require real and imaginary parts of the function as scipy.integrate.quad() supports real-valued functions only
                def f_r(x): return (1.0j * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - self.DG(np.array([nodes[i], 0]), np.array([x, 0]))).real
                def f_i(x): return (1.0j * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - self.DG(np.array([nodes[i], 0]), np.array([x, 0]))).imag

                for j in range(N):
                    if i != j:
                        A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(4*N))[0] + 1.0j * sci_int.quad(f_i, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(4*N))[0]
                    else:
                        # TODO: find analytical solution for i = j to avoid integrating over singularity
                        A[i, j] = diag
                print(A[i, i])
            B = np.identity(N) / 2 - A

        elif bc == "N":
            # neumann problem:
            #
            # WIP

            diag = - 1.0j * self.k ** 2 * (np.pi + 2.0j * (np.log(1 / N) + np.log(self.k / 2) + np.euler_gamma - 1)) / (2 * np.pi * N)

            for i in range(N):
                def f_r(x): return (1.0j * self.DG(np.array([x, 0]), np.array([nodes[i], 0])) - self.k ** 2 * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - self.D2G(np.array([nodes[i], 0]), np.array([x, 0]))).real
                def f_i(x): return (1.0j * self.DG(np.array([x, 0]), np.array([nodes[i], 0])) - self.k ** 2 * self.G(np.array([nodes[i], 0]), np.array([x, 0])) - self.D2G(np.array([nodes[i], 0]), np.array([x, 0]))).imag
                for j in range(N):
                    if i != j:
                        A[i, j] = sci_int.quad(f_r, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(4*N))[0] + 1.0j*sci_int.quad(f_i, nodes[j] - 1/N, nodes[j] + 1/N, limit=np.floor(4*N))[0]
                    else:
                        # TODO: find analytical solution for i = j to avoid integrating over singularity
                        A[i, j] = diag
                print(A[i, i])
            B = 1.0j * np.identity(N) / 2 - A
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
            def f_r(x): return ((self.DG(r, np.array([x, 0])) - 1.0j * self.G(r, np.array([x, 0]))) * self.weights[i]).real
            def f_i(x): return ((self.DG(r, np.array([x, 0])) - 1.0j * self.G(r, np.array([x, 0]))) * self.weights[i]).imag
            sum += sci_int.quad(f_r, antinodes[i], antinodes[i+1], limit=100)[0] + 1.0j * sci_int.quad(f_i, antinodes[i], antinodes[i+1], limit=100)[0]

        return sum

    def plot_uscat(self, n=50, *, inp_range=(-3, 3), totalu=False):
        xvals = np.linspace(inp_range[0], inp_range[1], n)
        yvals = np.linspace(inp_range[0], inp_range[1], n)

        x, y = np.meshgrid(xvals, yvals)

        stack = np.column_stack((x.flatten(), y.flatten()))

        z_vals = np.ones(len(stack))

        print(z_vals)
        print(len(z_vals))

        print(self.weights)

        for i in range(len(z_vals)):
            if totalu:
                z_vals[i] = (self.u_scat(stack[i]) + np.exp(1.0j * self.k * stack[i][1])).real
            else:
                z_vals[i] = self.u_scat(stack[i]).real
            if i % 100 == 0:
                print(f"{i}/{len(z_vals)}")

        z = np.reshape(z_vals, (-1, n)) 

        s = mlab.surf(x.T, y.T, z.T)

        mlab.show()

    def amplitude_sample(self, n=1000, r=10**7, *, absolute=False):
        x_vals = np.linspace(0, 2*np.pi, n, endpoint=False)
        y_vals = np.ones(n)
        for i in range(len(y_vals)):
            pos = (r*np.cos(x_vals[i]), r*np.sin(x_vals[i]))

            if absolute:
                y_vals[i] = abs(self.u_scat(pos) * np.sqrt(r) / np.exp(1.0j * self.k * r))
            else:
                y_vals[i] = (self.u_scat(pos) * np.sqrt(r) / np.exp(1.0j * self.k * r)).real

            if i % 100 == 0:
                print(f"{i}/{n}")

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, linewidth=2)
        ax.set_xlabel("Angle from center θ", fontsize=15)
        ax.set_ylabel("Absolute value of amplitude function |A(θ)|", fontsize=15)

        ticks = np.linspace(0, 2*np.pi, 5)
        xlabels = ["0", "π/2", "π", "3π/2", "2π"]
        ax.set_xticks(ticks, labels=xlabels)

        plt.show()

    def edge_condition_plot(self, n=1000):
        k = self.k
        b = (0, -0.5)[self.bcs == "N"]
        x_vals = [k**(-8*i/n + 2) for i in range(n)]

        filtered = [x for x in x_vals if x >= 10**(-7) and x <= 10*3]

        u_s = [abs(self.u_scat(np.array([1+x, -x])/np.sqrt(2))) for x in filtered]

        bound = [x**b for x in filtered]
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.plot(filtered, u_s, '-', label=r"$|u^{(s)}|$")
        ax.plot(filtered, bound, '--', label="Bounding line")
        ax.legend()
        plt.show()


########################
## TESTING & PLOTTING ##
########################

k = 10.00
# wave number

sys = HelmholtzSystem(k, "D")

# sys.plot_uscat(200, totalu=True)

sys.edge_condition_plot()

# sys.amplitude_sample(absolute=True)
