"""Generate useful visuals for the project."""

import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as ml
from ..helper.green_function import GreensFunctionCalculator
from ..helper.wave_components import u_d, u_r, u_i   # NOQA F401


def main():
    """Generate visuals with some defined constants."""
    WAVENUMBER = 10
    GRID_SIZE = 1000
    SOURCE_POINT = np.array([0, 0])

    # To display the schematic of the boundary conditions:
    # plot_boundary_schematic()

    plot_analytical_asymptotics()

    return

    # To display the 3D surface of the Green's function:
    plot_green_function_surface(WAVENUMBER, GRID_SIZE, r0=SOURCE_POINT)


def plot_boundary_schematic():
    """Plot a schematic of the boundary conditions using matplotlib."""
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\tilde{x}$", fontsize=10)
    ax.set_ylabel(r"$\tilde{y}$", fontsize=10)
    ax.plot([-3, 0], [0, 0], 'r-', linewidth=3,
            label=r'Boundary $\gamma$')
    ax.plot([0, 0], [-3, 3], '--', color="black", linewidth=2)
    ax.set_xbound(-3, 3)
    ax.set_ybound(-3, 3)
    ax.set_box_aspect(1)
    ax.text(-1.5, -1.5, '1', fontsize=20)
    ax.text(-1.5, 1.5, '2', fontsize=20)
    ax.text(1.5, -0.1, '3', fontsize=20)
    ax.legend()
    plt.show()


def analytical_solution(x, y, k):
    """Compute the analytical solution."""
    # if x < -1 or x > 1:
    #     return u_i(x, y, k) + u_d(x, y, k, -1) + u_d(x, y, k, 1)
    # elif y > 0:
    #     return u_d(x, y, k, -1) + u_d(x, y, k, 1)
    # else:
    #     return u_i(x, y, k) + u_r(x, y, k) + u_d(x, y, k, -1) + u_d(x, y, k, 1)

    return u_d(x, y, k, -1) + u_d(x, y, k, 1)


def plot_analytical_asymptotics():
    """Plot the analytical asymptotics."""
    r = 10**7
    n = 1000
    k = 50
    x_vals = np.linspace(0, 2*np.pi, n, endpoint=False)
    y_vals = np.ones(n)
    for i in range(len(y_vals)):
        pos = (r*np.cos(x_vals[i]), r*np.sin(x_vals[i]))

        y_vals[i] = abs((analytical_solution(pos[0], pos[1], k)) * np.sqrt(r) /
                        np.exp(1.0j * k * r))

        if i % 100 == 0:
            print(f"Calculating amplitude sample: {i}/{n} points completed.")

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, linewidth=2)
    ax.set_xlabel("Angle from center θ", fontsize=15)
    ax.set_ylabel("Absolute value of amplitude function |A(θ)|",
                  fontsize=15)

    ticks = np.linspace(0, 2*np.pi, 5)
    xlabels = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(ticks, labels=xlabels)

    plt.show()


def plot_analytical_solution():
    """Plot the analytical solution."""
    k = 10
    n = 1000

    xvals, yvals = np.linspace(-3, 3, n), np.linspace(-3, 3, n)
    x, y = np.meshgrid(xvals, yvals)

    z = np.ones(n**2)

    stack = np.vstack([x.ravel(), y.ravel()]).T

    for i in range(n**2):

        if i % 100 == 0:
            print(f"{i}/{n**2}")

        x_i, y_i = stack[i]
        # if x_i < -1 or x_i > 1:
        #     z[i] = u_i(x_i, y_i) + u_d(x_i, y_i, -1) + u_d(x_i, y_i, 1)
        # elif y_i > 0:
        #     z[i] = u_d(x_i, y_i, -1) + u_d(x_i, y_i, 1)
        # else:
        #     z[i] = u_i(x_i, y_i) + u_r(x_i, y_i) +
        #            u_d(x_i, y_i, -1) + u_d(x_i, y_i, 1)

        z[i] = u_d(x_i, y_i, k, -1) + u_d(x_i, y_i, k, 1)

        z[i] = z[i].real

    z_vals = z.reshape(x.shape)

    ml.surf(x.T, y.T, z_vals.T)
    ml.show()


def plot_green_function_surface(k, n, r0):
    """Plot the 3D surface of the Green's function using Mayavi."""
    print("Generating surface data...")
    greens_func = GreensFunctionCalculator(k, n)
    x, y, z = greens_func.generate_surface_data(r0)
    print("Data generated. Rendering surface...")
    ml.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    ml.surf(x.T, y.T, z.T, colormap='viridis')
    ml.xlabel("x")
    ml.ylabel("y")
    ml.zlabel("Re(G)")
    ml.show()
    print("Render complete.")


if __name__ == '__main__':
    main()
