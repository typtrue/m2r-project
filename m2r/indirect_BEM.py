import numpy as np
from scipy.special import hankel1
from math import e, cos, sin


class IndirectBEM:
    def __init__(self, interval, k=10, n=100):
        self.k = k
        self.phi = []
        self.n = min(n, 10*k + 2)  # number of elements
        self.interval = interval  # [-1, 1]
        self.intervals = interval_creator(interval[0], interval[1], n, k)


def interval_creator(lower, upper, n, k):
    """5k in each of the 1/k intervals either side of the endpoints."""
    if n == 0:
        return []

    n_boundary_region1 = max(0, int(round(5 * k)))
    n_boundary_region3 = max(0, int(round(5 * k)))
    n_middle_region = n - n_boundary_region1 - n_boundary_region3

    pt_A = float(lower)
    pt_D = float(upper)

    ideal_transition_B = pt_A + 1.0 / k
    ideal_transition_C = pt_D - 1.0 / k

    actual_B_endpoint = min(ideal_transition_B, pt_D)
    actual_B_endpoint = max(actual_B_endpoint, pt_A)

    actual_C_startpoint = max(ideal_transition_C, pt_A)
    actual_C_startpoint = min(actual_C_startpoint, pt_D)
    actual_C_startpoint = max(actual_C_startpoint, actual_B_endpoint)

    nodes_to_concatenate = []

    nodes_to_concatenate.append(
        np.linspace(pt_A, actual_B_endpoint, n_boundary_region1 + 1)
    )

    if actual_C_startpoint > actual_B_endpoint:
        nodes_to_concatenate.append(
            np.linspace(actual_B_endpoint, actual_C_startpoint, n_middle_region + 1)
        )

    nodes_to_concatenate.append(
        np.linspace(actual_C_startpoint, pt_D, n_boundary_region3 + 1)
    )

    final_nodes = np.unique(np.concatenate(nodes_to_concatenate))

    intervals = []
    for i in range(len(final_nodes) - 1):
        intervals.append((final_nodes[i], final_nodes[i+1]))

    return intervals


def green_function(self, x, y1, k):
    x1, x2 = x
    R = ((x1 - y1) + x2**2) ** (1/2)
    return -(1.0j)/4 * hankel1(0, k * R)


def kernel(x1, y1, k, n_x1):
    eps = 10**(-3)
    if x1 != y1:
        return 0
    


def incident_field(x1, x2, k, alpha):
    return e ** (1.0j * k * (x1 * cos(alpha) + x2 * sin(alpha)))


def trapezium_rule(n, x0, xn, f):
    xs = np.linspace(x0, xn, n)
    samples = np.array([f(x) for x in xs])
    h = (xn - x0) / n
    slice = np.array(samples[1:-1]) * 2
    samples[1:-1] = slice
    return 0.5*h*sum(samples)
