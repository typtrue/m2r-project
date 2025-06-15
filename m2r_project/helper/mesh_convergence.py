"""Mesh convergence test to verify accuracy of indirect BEM."""

import numpy as np


def mesh_convergence(n_values, metric_angle):
    """Mesh convergence study for the BEM solver and plots the results."""
    from ..BEM.indirect_BEM_combined import IndirectBEM

    print("Starting BEM Mesh Convergence Test.")

    test_params = {
        "intervals": [([-1, 0], [1, 0])],
        "alpha_rad": np.pi / 2,
        "k": 5.0
    }

    amplitude_results = []
    for n in n_values:
        print(f"  Calculating for n = {n} elements...")
        bem = IndirectBEM(
            intervals=test_params["intervals"],
            alpha=test_params["alpha_rad"],
            k=test_params["k"],
            n=n
        )
        amplitude = get_far_field_amplitude(bem, metric_angle)
        amplitude_results.append(np.abs(amplitude))

    return amplitude_results


def get_far_field_amplitude(bem, theta, r=1e7):
    """Calculate the far-field amplitude A(theta) for a given BEM solution."""
    pos = (r * np.cos(theta), r * np.sin(theta))
    u_scat_far = bem.calc_u_scat(np.array([pos]))[0]

    # Normalize to get the amplitude A(theta) using the far-field form
    # u_scat(r,θ) ≈ A(θ) * exp(ikr) / sqrt(r)
    amplitude = u_scat_far * np.sqrt(r) / np.exp(1.0j * bem.k * r)

    return amplitude
