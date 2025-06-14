from scipy.special import hankel1
import numpy as np


class GreensFunctionCalculator:
    """
    Calculates and visualizes the Green's function for the 2D Helmholtz eq.

    Attributes:
        k (int): The wavenumber for the calculations.
        n (int): The number of points for the grid.
    """

    def __init__(self, k, n=1000):
        """
        Initialize the calculator with a wavenumber and grid size.

        Args:
            k (int): The wavenumber.
            n (int): The number of points in each dimension of the grid.
        """
        self.k = k
        self.n = n

    @staticmethod
    def calculate_helmholtz_green_function(r, r0, k):
        """
        Calculate the Green's function for the 2D Helmholtz equation.

        Args:
            r (np.ndarray): The observation point.
            r0 (np.ndarray): The source point.
            k (float): The wavenumber.

        Returns:
            complex: The value of the Green's function.
        """
        distance = np.linalg.norm(r - r0)
        return 1.0j * hankel1(0, k * distance) / 4

    def calculate_partial_derivative_y(self, r, r0):
        """
        Calculate the partial derivative of the Green's function.

        This is with respect to the second coordinate of the second variable.

        Args:
            r (np.ndarray): The observation point (w, x).
            r0 (np.ndarray): The source point (y, z).

        Returns:
            complex: The value of the partial derivative.
        """
        distance = np.linalg.norm(r - r0)
        # Avoid division by zero if r and r0 are the same point.
        if distance == 0:
            return 0
        _, x = r
        _, z = r0
        factor = self.k * (x - z) / (4 * distance)
        return 1.0j * factor * hankel1(1, self.k * distance)

    def generate_surface_data(self, r0):
        """
        Generate the z-values for the Green's function surface plot.

        Args:
            r0 (np.ndarray): The source point.

        Returns:
            np.ndarray: A 2D array of the real part of the Green's function.
        """
        x_vals = np.linspace(-3, 3, self.n)
        y_vals = np.linspace(-3, 3, self.n)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        # Create a combined array of (x, y) coordinates
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))

        # Vectorized calculation
        distances = np.linalg.norm(grid_points - r0, axis=1)
        z_vals = 1.0j * hankel1(0, self.k * distances) / 4

        # Cap the values to avoid extreme peaks in the visualization
        z_vals[np.abs(z_vals) >= 5] = np.nan

        return x_grid, y_grid, z_vals.real.reshape(self.n, self.n)
