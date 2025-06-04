import numpy as np
from scipy.special import hankel1
from math import e, cos, sin
from scipy.integrate import quad
import matplotlib.pyplot as plt


class IndirectBEM:
    def __init__(self, start_point, end_point, alpha, k=10, n=100):
        self.k = k
        self.N = int(min(n, 10*k + 2))  # number of elements
        self.start_point = np.array(start_point)  # [x1_start, x2_start]
        self.end_point = np.array(end_point)      # [x1_end, x2_end]
        self.alpha = alpha
        self.offset_distance = 0.1 * np.linalg.norm(self.end_point - self.start_point) / self.N

        # Calculate line properties
        self.line_vector = self.end_point - self.start_point
        self.line_length = np.linalg.norm(self.line_vector)
        self.tangent = self.line_vector / self.line_length if self.line_length > 0 else np.array([1, 0])
        # Normal vector (rotate tangent 90 degrees counterclockwise)
        self.normal = np.array([-self.tangent[1], self.tangent[0]])

        self.interval_creator()
        self.calc_physical_mids()
        self.A = np.zeros((self.N, self.N), dtype=complex)
        self.g_prime = np.zeros(self.N, dtype=complex)
        self.phi = np.zeros(self.N, dtype=complex)
        self.calc_A()
        print(f"Condition number of A: {np.linalg.cond(self.A)}") 
        self.calc_g_prime()
        self.calc_phi()
        if self.phi is not None:
            print(f"Debug: Solved phi (first 5 elements): {self.phi[:5]}")
            print(f"Debug: Solved phi magnitude (mean): {np.mean(np.abs(self.phi))}")
            print(f"Debug: Solved phi magnitude (max): {np.max(np.abs(self.phi))}")
            print(f"Debug: Solved phi magnitude (min): {np.min(np.abs(self.phi))}")

    def param_to_physical(self, t):
        """Convert parameter t ∈ [0, 1] to physical coordinates on the line segment."""
        return self.start_point + t * self.line_vector

    def param_to_aux_physical(self, t):
        """Convert parameter t ∈ [0, 1] to physical coordinates on the auxiliary line."""
        return self.param_to_physical(t) - self.offset_distance * self.normal

    def physical_to_param(self, point):
        """Convert physical point to parameter t along the line segment."""
        if self.line_length == 0:
            return 0
        return np.dot(point - self.start_point, self.line_vector) / (self.line_length**2)

    def interval_creator(self):
        """Create parameter intervals in [0, 1] for the line segment."""
        if self.N == 0:
            self.param_intervals = []
            return

        n_boundary_region1 = max(0, int(round(5 * self.k)))
        n_boundary_region3 = max(0, int(round(5 * self.k)))
        n_middle_region = max(0, int(self.N - n_boundary_region1 - n_boundary_region3))

        # Parameter space is [0, 1]
        pt_A = 0.0
        pt_D = 1.0

        # Scale the 1/k regions to parameter space
        param_scale = 1.0 / (self.k * self.line_length) if self.line_length > 0 else 0.1
        ideal_transition_B = pt_A + param_scale
        ideal_transition_C = pt_D - param_scale

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
            self.param_intervals = np.unique(np.concatenate(nodes_to_concatenate))
        else:
            self.param_intervals = np.array([pt_A, pt_D])

    def calc_physical_mids(self):
        """Calculate physical midpoints of each interval."""
        param_mids = np.array([(self.param_intervals[i] + self.param_intervals[i+1])/2
                              for i in range(len(self.param_intervals)-1)])
        self.mids = np.array([self.param_to_physical(t) for t in param_mids])

    def kernel(self, x, y, normal_vec):
        """Kernel function for points in 2D space."""
        diff = x - y
        R = np.linalg.norm(diff)

        if np.isclose(R, 0):
            return 0 + 0j

        # Direction from y to x
        direction = diff / R
        # Normal derivative
        normal_deriv = np.dot(direction, normal_vec)

        return normal_deriv * (1j * self.k / 4.0) * hankel1(1, self.k * R)

    def incident_field(self, x1, x2, alpha):
        return e ** (1j * self.k * (x1 * cos(alpha) + x2 * sin(alpha)))

    def calc_g_prime(self):
        """Calculate the boundary condition g' = -∂u_inc/∂n."""
        u_inc = np.array([self.incident_field(mid[0], mid[1], self.alpha) for mid in self.mids])

        # Gradient of incident field
        duinc_dx1 = 1j * self.k * cos(self.alpha) * u_inc
        duinc_dx2 = 1j * self.k * sin(self.alpha) * u_inc

        # Normal derivative
        self.g_prime = -(self.normal[0] * duinc_dx1 + self.normal[1] * duinc_dx2)

    def f_y_param(self, t, x, normal_vec):
        """Integrand function in parameter space."""
        y = self.param_to_physical(t)
        diff = x - y
        R = np.linalg.norm(diff)

        if np.isclose(R, 0.0):
            # Handle singularity
            val = -np.dot(normal_vec, normal_vec) * (1j * self.k / 4.0) * (-2j / np.pi)
            return (val.real + val.imag * 1j) * self.line_length

        direction = diff / R
        normal_deriv = np.dot(direction, normal_vec)
        integrand_val = -normal_deriv * (1j * self.k / 4.0) * R * hankel1(1, self.k * R)

        # Multiply by Jacobian (line_length) for parameter transformation
        return integrand_val * self.line_length

    def calc_A(self):
        """Calculate the matrix A."""
        for i in range(len(self.mids)):
            x_collocation_point = self.mids[i]
            normal_at_x_collocation = self.normal

            for j in range(len(self.mids)):
                t_a_j = self.param_intervals[j]
                t_b_j = self.param_intervals[j+1]

                def kernel_real_param_aux(t_param_source, x_coll_pt, normal_at_x_coll_pt):
                    y_source_pt_aux = self.param_to_aux_physical(t_param_source)
                    return np.real(self.kernel(x_coll_pt, y_source_pt_aux, normal_at_x_coll_pt)) * self.line_length

                def kernel_imag_param_aux(t_param_source, x_coll_pt, normal_at_x_coll_pt):
                    y_source_pt_aux = self.param_to_aux_physical(t_param_source)
                    return np.imag(self.kernel(x_coll_pt, y_source_pt_aux, normal_at_x_coll_pt)) * self.line_length

                # Numerical integration over the j-th source element on Gamma_aux
                real_part, _ = quad(kernel_real_param_aux, t_a_j, t_b_j,
                                    args=(x_collocation_point, normal_at_x_collocation))
                imag_part, _ = quad(kernel_imag_param_aux, t_a_j, t_b_j,
                                    args=(x_collocation_point, normal_at_x_collocation))

                self.A[i, j] = real_part + 1j * imag_part

    def calc_phi(self):
        try:
            self.phi = np.linalg.solve(self.A, self.g_prime)
        except np.linalg.LinAlgError as e:
            print(f"Error solving linear system: {e}")
            return None

    def green_function(self, x, y):
        """Green's function between two 2D points."""
        diff = np.array(x) - np.array(y)
        R = np.linalg.norm(diff)

        if np.isclose(R, 0.0):
            return np.inf + 0.0j
        return -(1j)/4 * hankel1(0, self.k * R)

    def calc_u_scat(self, x):
        """Calculate scattered field at evaluation points x."""
        u_scattered = np.zeros(len(x), dtype=complex)

        for idx_x, x in enumerate(x):
            if idx_x % 100 == 0:
                print(f"{idx_x} completed.")
            val_at_x = 0.0 + 0.0j

            for j in range(len(self.mids)):
                phi_j = self.phi[j]
                t_a = self.param_intervals[j]
                t_b = self.param_intervals[j+1]

                def green_real_param(t, x_pt):
                    y_pt = self.param_to_aux_physical(t)
                    return np.real(self.green_function(x_pt, y_pt)) * self.line_length

                def green_imag_param(t, x_pt):
                    y_pt = self.param_to_aux_physical(t)
                    return np.imag(self.green_function(x_pt, y_pt)) * self.line_length

                real_part, _ = quad(green_real_param, t_a, t_b, args=(x,))
                imag_part, _ = quad(green_imag_param, t_a, t_b, args=(x,))

                integral_G_dy = real_part + 1j * imag_part
                val_at_x += phi_j * integral_G_dy

            u_scattered[idx_x] = val_at_x

        return u_scattered


def run_bem_test(start_point, end_point, alpha_rad, k, n):
    alpha_deg = np.rad2deg(alpha_rad)
    bem = IndirectBEM(start_point=start_point, end_point=end_point, alpha=alpha_rad, k=k, n=n)

    # Density plots (phi)
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'k={k}, normal=[{bem.normal[0]:.2f},{bem.normal[1]:.2f}], No. Elements={bem.N}, Angle={alpha_deg}°')

    # Parameter values for plotting
    param_mids = np.array([(bem.param_intervals[i] + bem.param_intervals[i+1])/2 
                          for i in range(len(bem.param_intervals)-1)])

    plt.subplot(1, 2, 1)
    plt.plot(param_mids, np.real(bem.phi), 'b.-', label='Re($\phi$)')
    plt.plot(param_mids, np.imag(bem.phi), 'r.-', label='Im($\phi$)')
    plt.xlabel('Parameter $t$ along boundary')
    plt.ylabel('Density $\phi(t)$')
    plt.legend()
    plt.grid(True)
    plt.title('Solved Density $\phi$')

    plt.subplot(1, 2, 2)
    plt.plot(param_mids, np.abs(bem.phi), 'g.-', label='$|\phi|$')
    plt.xlabel('Parameter $t$ along boundary')
    plt.ylabel('Magnitude $|\phi(t)|$')
    plt.legend()
    plt.grid(True)
    plt.title('Magnitude of $\phi$')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Field plots
    grid_res = 60
    # Determine plot bounds based on line segment
    all_coords = np.vstack([bem.start_point, bem.end_point])
    x_min, x_max = all_coords[:, 0].min() - 2, all_coords[:, 0].max() + 2
    y_min, y_max = all_coords[:, 1].min() - 2, all_coords[:, 1].max() + 2

    x_coords = np.linspace(x_min, x_max, grid_res)
    y_coords = np.linspace(y_min, y_max, grid_res)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    eval_points_grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    u_inc_grid = bem.incident_field(eval_points_grid[:, 0], eval_points_grid[:, 1], alpha_rad)

    print(f"Calculating scattered field on a {grid_res}x{grid_res} grid...")
    u_scat_grid = bem.calc_u_scat(eval_points_grid)
    U_scat_magnitude = np.abs(u_scat_grid).reshape(X_grid.shape)

    U_total_magnitude = np.abs(u_inc_grid + u_scat_grid).reshape(X_grid.shape)

    plt.figure(figsize=(21, 6))
    plt.suptitle(f'Wave Fields (k={k}, Line from {start_point} to {end_point}, Angle={alpha_deg}°)', fontsize=14)

    # Subplot 1: Incident Wave Path (Arrows) + Boundary
    ax1 = plt.subplot(1, 3, 1)
    arrow_skip = max(1, grid_res // 10)
    X_q, Y_q = X_grid[::arrow_skip, ::arrow_skip], Y_grid[::arrow_skip, ::arrow_skip]

    u_direction = np.cos(alpha_rad)
    v_direction = np.sin(alpha_rad)
    U_q = np.full_like(X_q, u_direction)
    V_q = np.full_like(Y_q, v_direction)

    sine_alpha = np.sin(alpha_rad)
    if sine_alpha > 1e-6:
        mask = Y_q > max(bem.start_point[1], bem.end_point[1])
        U_q[mask] = np.nan
        V_q[mask] = np.nan
    elif sine_alpha < -1e-6:
        mask = Y_q < min(bem.start_point[1], bem.end_point[1])
        U_q[mask] = np.nan
        V_q[mask] = np.nan

    ax1.quiver(X_q, Y_q, U_q, V_q, angles='xy', scale_units='xy', scale=(k/1.5 or 5), 
               color='cyan', headwidth=5, headlength=7, width=0.004, pivot='tail')
    ax1.plot([bem.start_point[0], bem.end_point[0]], [bem.start_point[1], bem.end_point[1]], 
             'r-', linewidth=3, label='Boundary $\Gamma$')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('Incident Wave Path')
    ax1.legend()
    ax1.set_xlim(x_coords.min(), x_coords.max())
    ax1.set_ylim(y_coords.min(), y_coords.max())
    ax1.set_aspect('equal', adjustable='box')

    # Subplot 2: Scattered Field Magnitude
    ax2 = plt.subplot(1, 3, 2)
    im_scat = ax2.imshow(U_scat_magnitude, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                         origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im_scat, ax=ax2, label='$|u_{scat}|$')
    ax2.plot([bem.start_point[0], bem.end_point[0]], [bem.start_point[1], bem.end_point[1]], 
             'r-', linewidth=3, label='Boundary $\Gamma$')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Scattered Field Magnitude $|u_{scat}|$')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')

    # Subplot 3: Total Field Magnitude
    ax3 = plt.subplot(1, 3, 3)
    im_total = ax3.imshow(U_total_magnitude, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                          origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im_total, ax=ax3, label='$|u_{total}|$')
    ax3.plot([bem.start_point[0], bem.end_point[0]], [bem.start_point[1], bem.end_point[1]], 
             'r-', linewidth=3, label='Boundary $\Gamma$')
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title('Total Field Magnitude $|u_{inc} + u_{scat}|$')
    ax3.legend()
    ax3.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("-" * 70 + "\n")
    return bem


if __name__ == '__main__':
    # Vertical line
    # run_bem_test([-1, 0], [1, 0], np.pi/2, 8.0, 160)

    # Diagonal line
    run_bem_test([-1, 1], [1, -1], np.pi/6, 10.0, 200)
