"""Generate plots and visuals."""

import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as ml
from helper.green_function import GreensFunctionCalculator


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


def plot_green_function_surface(k, n, r0):
    """
    Plot the 3D surface of the Green's function using Mayavi.

    Args:
        k (int): The wavenumber for the calculations.
        n (int): The number of points for the grid.
        r0 (np.ndarray): The source point for the function.
    """
    print("Generating surface data...")
    greens_function = GreensFunctionCalculator(k, n)
    x, y, z = greens_function.generate_surface_data(r0)
    print("Data generated. Rendering surface...")
    ml.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    ml.surf(x.T, y.T, z.T, colormap='viridis')
    ml.xlabel("x")
    ml.ylabel("y")
    ml.zlabel("Re(G)")
    ml.show()
    print("Render complete.")


if __name__ == '__main__':
    WAVENUMBER = 10
    GRID_SIZE = 1000
    SOURCE_POINT = np.array([0, 0])

    # To display the schematic of the boundary conditions:
    plot_boundary_schematic()

    # To display the 3D surface of the Green's function:
    plot_green_function_surface(WAVENUMBER, GRID_SIZE, r0=SOURCE_POINT)
