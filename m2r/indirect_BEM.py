import numpy as np
from scipy.special import hankel1
from math import e, cos, sin
from scipy.integrate import quad
import matplotlib.pyplot as plt


class IndirectBEM:
    def __init__(self, interval, alpha, normal, k=10, n=100):
        self.k = k
        self.phi = []
        self.N = int(min(n, 10*k + 2))  # number of elements
        self.interval = interval  # [-1, 1]
        self.alpha = alpha
        self.normal = normal
        self.interval_creator()
        self.mids = np.array([(self.intervals[i] + self.intervals[i+1])/2 for i in range(len(self.intervals)-1)])
        self.A = np.zeros((self.N, self.N), dtype=complex)
        self.g_prime = np.zeros(self.N, dtype=complex)
        self.phi = np.zeros(self.N, dtype=complex)
        self.calc_A()
        self.calc_g_prime()
        self.calc_phi()

    def interval_creator(self):
        """5k in each of the 1/k intervals either side of the endpoints."""
        if self.N == 0:
            self.intervals = []
            return

        n_boundary_region1 = max(0, int(round(5 * self.k)))
        n_boundary_region3 = max(0, int(round(5 * self.k)))
        n_middle_region = max(0, int(self.N - n_boundary_region1 - n_boundary_region3))

        pt_A = float(self.interval[0])
        pt_D = float(self.interval[1])

        ideal_transition_B = pt_A + 1.0 / self.k
        ideal_transition_C = pt_D - 1.0 / self.k

        actual_B_endpoint = min(ideal_transition_B, pt_D)
        actual_B_endpoint = max(actual_B_endpoint, pt_A)

        actual_C_startpoint = max(ideal_transition_C, pt_A)
        actual_C_startpoint = min(actual_C_startpoint, pt_D)
        actual_C_startpoint = max(actual_C_startpoint, actual_B_endpoint)

        nodes_to_concatenate = []

        if n_boundary_region1 > 0:
            nodes_to_concatenate.append(
                np.linspace(pt_A, actual_B_endpoint, n_boundary_region1 + 1)
            )

        if actual_C_startpoint > actual_B_endpoint and n_middle_region > 0:
            nodes_to_concatenate.append(
                np.linspace(actual_B_endpoint, actual_C_startpoint, n_middle_region + 1)
            )

        if n_boundary_region3 > 0:
            nodes_to_concatenate.append(
                np.linspace(actual_C_startpoint, pt_D, n_boundary_region3 + 1)
            )

        if nodes_to_concatenate:
            self.intervals = np.unique(np.concatenate(nodes_to_concatenate))
        else:
            self.intervals = np.array([pt_A, pt_D])

    def kernel(self, x1, y1, n1):
        if np.isclose(x1, y1):
            return 0
        elif np.isclose(n1, 0):
            return 0 + 0j
        S = np.sign(x1 - y1)
        R = np.abs(x1 - y1)
        return n1 * (1j * self.k / 4.0) * S * hankel1(1, self.k * R)

    def incident_field(self, x1, x2, alpha):
        return e ** (1j * self.k * (x1 * cos(alpha) + x2 * sin(alpha)))

    def calc_g_prime(self):
        u_inc = np.exp(1j * self.k * self.mids * np.cos(self.alpha))
        duinc_dx1 = 1j * self.k * cos(self.alpha) * u_inc
        duinc_dx2 = 1j * self.k * sin(self.alpha) * u_inc
        self.g_prime = -(self.normal[0] * duinc_dx1 + self.normal[1] * duinc_dx2)

    def f_y1(self, y1, x1, n1):
        if np.isclose(n1, 0.0):
            return 0.0 + 0.0j

        R = np.abs(x1 - y1)

        if np.isclose(R, 0.0):
            val = -n1 * (1j * self.k / 4.0) * (-2j / np.pi)
            return val.real + val.imag * 1j

        integrand_val = -n1 * (1j * self.k / 4.0) * R * hankel1(1, self.k * R)
        return integrand_val

    def calc_A(self):
        if np.isclose(self.normal[0], 0.0):
            np.fill_diagonal(self.A, 0.5 + 0.0j)
            return

        for i in range(len(self.mids)):
            a_i = self.intervals[i]
            b_i = self.intervals[i+1]

            # Handle principal value integral by splitting into real and imaginary parts
            def f_y1_real(y1, x1, n1):
                return np.real(self.f_y1(y1, x1, n1))
            
            def f_y1_imag(y1, x1, n1):
                return np.imag(self.f_y1(y1, x1, n1))

            args_pv = (self.mids[i], self.normal[0])

            pv_real, _ = quad(f_y1_real, a_i, b_i, args=args_pv, weight='cauchy', wvar=self.mids[i])
            pv_imag, _ = quad(f_y1_imag, a_i, b_i, args=args_pv, weight='cauchy', wvar=self.mids[i])
            
            self.A[i, i] = 0.5 + pv_real + 1j * pv_imag

            for j in range(len(self.mids)):
                if i == j:
                    continue

                def kernel_real(y1_s, x1_f, n1_f):
                    return np.real(self.kernel(x1_f, y1_s, n1_f))
                
                def kernel_imag(y1_s, x1_f, n1_f):
                    return np.imag(self.kernel(x1_f, y1_s, n1_f))

                real_part, _ = quad(kernel_real, self.intervals[j], self.intervals[j+1], args=(self.mids[i], self.normal[0]))
                imag_part, _ = quad(kernel_imag, self.intervals[j], self.intervals[j+1], args=(self.mids[i], self.normal[0]))

                self.A[i, j] = real_part + 1j * imag_part

    def calc_phi(self):
        try:
            self.phi = np.linalg.solve(self.A, self.g_prime)
            return self.phi
        except np.linalg.LinAlgError as e:
            print(f"Error solving linear system: {e}")
            return None

    def green_function(self, x, y1):
        x1, x2 = x
        R = np.sqrt((x1 - y1)**2 + (x2 - 0.0)**2)

        if np.isclose(R, 0.0):
            return np.inf + 0.0j
        return -(1j)/4 * hankel1(0, self.k * R)

    def calc_u_scat(self, x):
        u_scattered = np.zeros(len(x), dtype=complex)

        for idx_x, x_point in enumerate(x):
            if idx_x % 100 == 0:
                print(f"{idx_x} completed.")
            val_at_x_point = 0.0 + 0.0j
            for j in range(len(self.mids)):
                phi_j = self.phi[j]

                def green_real(y1_s, x_f):
                    return np.real(self.green_function(x_f, y1_s))
                
                def green_imag(y1_s, x_f):
                    return np.imag(self.green_function(x_f, y1_s))

                real_part, _ = quad(green_real, self.intervals[j], self.intervals[j+1], args=(x_point,))
                imag_part, _ = quad(green_imag, self.intervals[j], self.intervals[j+1], args=(x_point,))
                
                integral_G_dy1 = real_part + 1j * imag_part
                val_at_x_point += phi_j * integral_G_dy1
            u_scattered[idx_x] = val_at_x_point

        return u_scattered


def run_bem_test(bounds, alpha, normal_v, k, n):
    alpha_rad = np.deg2rad(alpha)
    bem = IndirectBEM(interval=bounds, alpha=alpha_rad, normal=normal_v, k=k, n=n)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f'k={k}, normal=[{normal_v[0]:.2f},{normal_v[1]:.2f}], No. Elements={bem.N}, Angle={alpha}°')

    plt.subplot(1, 2, 1)
    plt.plot(bem.mids, np.real(bem.phi), 'b.-', label='Re($\phi$)')
    plt.plot(bem.mids, np.imag(bem.phi), 'r.-', label='Im($\phi$)')
    plt.xlabel('$x_1$ on boundary $\Gamma$')
    plt.ylabel('Density $\phi(x_1)$')
    plt.legend()
    plt.grid(True)
    plt.title('Solved Density $\phi$')

    plt.subplot(1, 2, 2)
    plt.plot(bem.mids, np.abs(bem.phi), 'g.-', label='$|\phi|$')
    plt.xlabel('$x_1$ on boundary $\Gamma$')
    plt.ylabel('Magnitude $|\phi(x_1)|$')
    plt.legend()
    plt.grid(True)
    plt.title('Magnitude of $\phi$')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    grid_res = 50
    x_coords = np.linspace(min(bounds[0]-1, -2), max(bounds[1]+1, 2), grid_res)
    y_coords = np.linspace(-1, 2, grid_res)  # Start slightly above the boundary
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    eval_points_grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    print(f"  Calculating scattered field on a {grid_res}x{grid_res} grid (this might take a moment)...")
    u_scat_grid = bem.calc_u_scat(eval_points_grid)
    U_scat_magnitude = np.abs(u_scat_grid).reshape(X_grid.shape)

    # Total field
    u_inc_grid = bem.incident_field(eval_points_grid[:, 0], eval_points_grid[:, 1], alpha_rad)
    U_total_magnitude = np.abs(u_inc_grid + u_scat_grid).reshape(X_grid.shape)

    plt.figure(figsize=(14, 6))
    plt.suptitle(f'Wave Fields (k={k}, normal=[{normal_v[0]:.2f},{normal_v[1]:.2f}], Angle={alpha}°)', fontsize=14)

    plt.subplot(1, 2, 1)
    plt.imshow(U_scat_magnitude, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], 
               origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='$|u_{scat}|$')
    plt.plot(bounds, [0, 0], 'r-', linewidth=3, label='Boundary $\Gamma$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Scattered Field Magnitude $|u_{scat}|$')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(U_total_magnitude, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
               origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='$|u_{total}|$')
    plt.plot(bounds, [0, 0], 'r-', linewidth=3, label='Boundary $\Gamma$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Total Field Magnitude $|u_{inc} + u_{scat}|$')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("-" * 70 + "\n")
    return bem


if __name__ == '__main__':
    # run_bem_test([-1, 1], 0, [0, 1], 5.0, 100)  # parallel wave
    run_bem_test([-2, 2], np.pi/4, [1, 0], 8.0, 160)  # perpendicular wave
    run_bem_test([-1, 1], np.pi/6, [1/np.sqrt(2), 1/np.sqrt(2)], 10.0, 200)  # angled wave
