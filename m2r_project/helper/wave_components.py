"""Define the components of the wave."""
import numpy as np


def u_d(x, y, k, x_source):
    """Diffraction wave."""
    pi = np.pi
    if x_source == 1:
        diff = x - x_source
    else:
        diff = x_source - x
    norm = np.sqrt(diff**2 + y**2)
    s1 = 1.0j * np.exp(1.0j * pi / 4) / (np.sqrt(2*pi*k) * diff)
    s2 = np.sqrt(norm - diff)
    s3 = np.exp(1.0j * k * norm)
    return s1 * s2 * s3


def u_r(x, y, k):
    """Reflection wave."""
    return np.exp(1.0j * k * -y)


def u_i(x, y, k):
    """Incident wave."""
    return np.exp(1.0j * k * y)
