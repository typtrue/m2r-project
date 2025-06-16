"""Implements calculations related to Greens Function."""

from scipy.special import hankel1
import numpy as np


class GreensFunctionCalculator:
    """Calculates & visualizes the Green's function for the 2D Helmholtz eq."""

    def __init__(self, k, n=1000):
        """Initialize the calculator with a wavenumber and grid size."""
        self.k = k
        self.n = n

    def greens_func(self, r, r0):
        """Calculate the Green's function for the 2D Helmholtz equation."""
        if not isinstance(r, np.ndarray):
            r = np.array(r)
        if not isinstance(r0, np.ndarray):
            r0 = np.array(r0)

        R = np.linalg.norm(r - r0)
        return 1.0j * hankel1(0, self.k * R) / 4

    def dir_deriv(self, x, y, unit_vec):
        """Partial derivative of the Green's function w.r.t. unit vector."""
        diff = x - y
        R = np.linalg.norm(diff)
        direction = diff / R
        unit_deriv = np.dot(direction, unit_vec)
        return unit_deriv * (1j * self.k / 4.0) * hankel1(1, self.k * R)

    def mixed_dir_deriv(self, x, y, unit_vec_x, unit_vec_y):
        """Partial derivative of the Green's function w.r.t x and y."""
        diff = x - y
        R = np.linalg.norm(diff)
        h_0_2 = hankel1(0, self.k * R) - hankel1(2, self.k * R)
        h_1 = hankel1(1, self.k * R)
        s1 = 2 * np.dot(diff, unit_vec_x)**2 * h_1 * np.dot(unit_vec_x,
                                                            unit_vec_y)
        s2 = np.dot(diff, unit_vec_x) * R * np.dot(diff, unit_vec_y) * h_0_2
        s3 = -2 * np.dot(diff, unit_vec_x) * np.dot(diff, unit_vec_y) * h_1

        return (s1 + s2 + s3) / 2 * R**3

    def generate_surface_data(self, r0, cap_value=5):
        """Generate the z-values for the Green's function surface plot."""
        x_vals = np.linspace(-3, 3, self.n)
        y_vals = np.linspace(-3, 3, self.n)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        # Create a combined array of (x, y) coordinates
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))

        # Vectorized calculation
        distances = np.linalg.norm(grid_points - r0, axis=1)
        z_vals = 1.0j * hankel1(0, self.k * distances) / 4

        # Cap the values to avoid extreme peaks in the visualization
        z_vals[np.abs(z_vals) >= cap_value] = np.nan

        return x_grid, y_grid, z_vals.real.reshape(self.n, self.n)
