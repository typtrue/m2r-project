"""Generate visuals to test the convergence of indirect BEM algorithm."""
import numpy as np
import matplotlib.pyplot as plt
from ..helper.mesh_convergence import mesh_convergence


def main():
    """Set up and test BEM for different mesh discretisation."""
    n_values = [40, 80, 120, 160, 200, 240, 280]
    metric_angle = 3 * np.pi / 2
    amplitudes = mesh_convergence(n_values, metric_angle)
    direct_convergence(n_values, amplitudes, metric_angle)
    log_log_error(n_values, amplitudes)


def direct_convergence(n, amplitudes, metric_angle=3*np.pi/2):
    """Direct convergence of the amplitude value."""
    plt.figure(figsize=(12, 6))
    plt.plot(n, amplitudes, 'o-', markerfacecolor='cyan',
             markeredgecolor='k', color='k')
    plt.xlabel("Number of Boundary Elements (N)")
    plt.ylabel(f"|A(θ = {metric_angle/np.pi:.1f}π)|")
    plt.title("Amplitude vs. Mesh Refinement")
    plt.grid(True, linestyle='--', alpha=0.6)


def log_log_error(n, amplitudes):
    """Log log error plot of the amplitude value against finest mesh."""
    true_value = amplitudes[-1]
    errors = [np.abs(res - true_value) for res in amplitudes[:-1]]
    h_values = [1/n for n in n[:-1]]

    plt.figure(figsize=(12, 6))
    plt.loglog(h_values, errors, 's-', markerfacecolor='salmon',
               markeredgecolor='k', color='k')
    plt.xlabel("Mesh Size (h ~ 1/N)")
    plt.ylabel("Absolute Error |A_N - A_true|")
    plt.title("Log-Log Error Plot")
    plt.grid(True, which="both", linestyle='--', alpha=0.6)

    plt.suptitle("BEM Solver Mesh Convergence Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
