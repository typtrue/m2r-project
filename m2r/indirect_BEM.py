import numpy as np
from scipy.special import hankel1
from math import e, cos, sin
from scipy.integrate import quad
import matplotlib.pyplot as plt
import mayavi.mlab as ml


class IndirectBEM:
    def __init__(self, intervals, alpha, k=10, n=100):
        self.k = k
        self.alpha = alpha
        self.intervals = [(np.array(start), np.array(end)) for start, end in intervals]

        self.line_vectors = [end - start for start, end in self.intervals]
        self.line_lengths = [np.linalg.norm(vec) for vec in self.line_vectors]
        self.tangents = [vec / length if length > 0 else np.array([1, 0]) for vec, length in zip(self.line_vectors, self.line_lengths)]
        self.normals = [np.array([-tangent[1], tangent[0]]) for tangent in self.tangents]

        total_length = sum(self.line_lengths)
        self.Ns = [int(round(n * length / total_length)) if total_length > 0 else int(n / len(self.intervals)) for length in self.line_lengths]
        self.N = sum(self.Ns)
        self.offset_distances = [0.1 * length / N_i if N_i > 0 else 0 for length, N_i in zip(self.line_lengths, self.Ns)]

        self.interval_creator()
        self.calc_physical_mids()
        self.A = np.zeros((self.N, self.N), dtype=complex)
        self.g_prime = np.zeros(self.N, dtype=complex)
        self.phi = np.zeros(self.N, dtype=complex)
        self.calc_A()
        self.calc_g_prime()
        self.calc_phi()

    def param_to_physical(self, t, interval_idx):
        """Convert parameter t ∈ [0, 1] to physical coordinates on the line segment."""
        return self.intervals[interval_idx][0] + t * self.line_vectors[interval_idx]

    def param_to_aux_physical(self, t, interval_idx):
        """Convert parameter t ∈ [0, 1] to physical coordinates on the auxiliary line."""
        return self.param_to_physical(t, interval_idx) - self.offset_distances[interval_idx] * self.normals[interval_idx]

    def physical_to_param(self, point):
        """Convert physical point to parameter t along the line segment."""
        if self.line_length == 0:
            return 0
        return np.dot(point - self.start_point, self.line_vector) / (self.line_length**2)

    def interval_creator(self):
        """Create parameter intervals in [0, 1] for each line segment."""
        self.all_param_intervals = []
        for i in range(len(self.intervals)):
            if self.Ns[i] == 0:
                self.all_param_intervals.append([])
                continue

            target_n = self.Ns[i]
            n_boundary = int(round(5 * self.k))
            if target_n >= 2 * n_boundary:
                n_boundary_region1 = n_boundary
                n_boundary_region3 = n_boundary
                n_middle_region = target_n - n_boundary_region1 - n_boundary_region3
            else:
                n_boundary_region1 = target_n // 2
                n_boundary_region3 = target_n - n_boundary_region1
                n_middle_region = 0

            pt_A, pt_D = 0.0, 1.0
            param_scale = 1.0 / (self.k * self.line_lengths[i]) if self.line_lengths[i] > 0 else 0.1
            ideal_transition_B = pt_A + param_scale
            ideal_transition_C = pt_D - param_scale
            actual_B_endpoint = min(ideal_transition_B, pt_D)
            actual_B_endpoint = max(actual_B_endpoint, pt_A)
            actual_C_startpoint = max(ideal_transition_C, pt_A)
            actual_C_startpoint = min(actual_C_startpoint, pt_D)
            actual_C_startpoint = max(actual_C_startpoint, actual_B_endpoint)
            nodes_to_concatenate = []
            if n_boundary_region1 > 0:
                nodes_to_concatenate.append(np.linspace(pt_A, actual_B_endpoint, n_boundary_region1 + 1))
            if actual_C_startpoint > actual_B_endpoint and n_middle_region > 0:
                nodes_to_concatenate.append(np.linspace(actual_B_endpoint, actual_C_startpoint, n_middle_region + 1))
            if n_boundary_region3 > 0:
                # To handle the case where the middle region is skipped, ensure concatenation starts from the correct point.
                start_point_reg3 = actual_C_startpoint if (actual_C_startpoint > actual_B_endpoint and n_middle_region > 0) else actual_B_endpoint
                nodes_to_concatenate.append(np.linspace(start_point_reg3, pt_D, n_boundary_region3 + 1))

            if nodes_to_concatenate:
                self.all_param_intervals.append(np.unique(np.concatenate(nodes_to_concatenate)))
            else:
                self.all_param_intervals.append(np.array([pt_A, pt_D]))

    # Calculate midpoints across all intervals and store their properties.
    def calc_physical_mids(self):
        """Calculate physical midpoints of each element across all intervals."""
        self.mids = []
        self.mid_interval_indices = []
        self.element_param_bounds = []
        for i, param_intervals in enumerate(self.all_param_intervals):
            if len(param_intervals) < 2:
                continue
            for j in range(len(param_intervals) - 1):
                t_a, t_b = param_intervals[j], param_intervals[j + 1]
                param_mid = (t_a + t_b) / 2
                self.mids.append(self.param_to_physical(param_mid, i))
                self.mid_interval_indices.append(i)
                self.element_param_bounds.append((t_a, t_b))
        self.mids = np.array(self.mids)

    def kernel(self, x, y, normal_vec):
        """Kernel function for points in 2D space."""
        diff = x - y
        R = np.linalg.norm(diff)

        if np.isclose(R, 0):
            return 0 + 0j

        direction = diff / R
        normal_deriv = np.dot(direction, normal_vec)

        return normal_deriv * (1j * self.k / 4.0) * hankel1(1, self.k * R)

    def incident_field(self, x1, x2, alpha):
        return e ** (1j * self.k * (x1 * cos(alpha) + x2 * sin(alpha)))

    def calc_g_prime(self):
        """Calculate the boundary condition g' = -∂u_inc/∂n."""
        u_inc = np.array([self.incident_field(mid[0], mid[1], self.alpha) for mid in self.mids])

        duinc_dx1 = 1j * self.k * cos(self.alpha) * u_inc
        duinc_dx2 = 1j * self.k * sin(self.alpha) * u_inc

        # Normal derivative using per-element normals.
        normals_array = np.array([self.normals[i] for i in self.mid_interval_indices])
        self.g_prime = -(normals_array[:, 0] * duinc_dx1 + normals_array[:, 1] * duinc_dx2)

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
        for i in range(self.N):
            x_mid_point = self.mids[i]
            normal_at_x_mid = self.normals[self.mid_interval_indices[i]]

            for j in range(self.N):
                t_a_j, t_b_j = self.element_param_bounds[j]
                interval_idx_j = self.mid_interval_indices[j]
                line_length_j = self.line_lengths[interval_idx_j]

                def kernel_real_param_aux(t_param_source, x_coll_pt, normal_at_x_coll_pt, src_interval_idx):
                    y_source_pt_aux = self.param_to_aux_physical(t_param_source, src_interval_idx)
                    return np.real(self.kernel(x_coll_pt, y_source_pt_aux, normal_at_x_coll_pt)) * line_length_j

                def kernel_imag_param_aux(t_param_source, x_coll_pt, normal_at_x_coll_pt, src_interval_idx):
                    y_source_pt_aux = self.param_to_aux_physical(t_param_source, src_interval_idx)
                    return np.imag(self.kernel(x_coll_pt, y_source_pt_aux, normal_at_x_coll_pt)) * line_length_j

                # Numerical integration over the j-th source element on Gamma_aux
                real_part, _ = quad(kernel_real_param_aux, t_a_j, t_b_j,
                                    args=(x_mid_point, normal_at_x_mid, interval_idx_j))
                imag_part, _ = quad(kernel_imag_param_aux, t_a_j, t_b_j,
                                    args=(x_mid_point, normal_at_x_mid, interval_idx_j))

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

        for idx_x, x_eval in enumerate(x):
            if idx_x % 100 == 0:
                print(f"{idx_x} completed.")
            val_at_x = 0.0 + 0.0j

            for j in range(self.N):
                phi_j = self.phi[j]
                t_a, t_b = self.element_param_bounds[j]
                interval_idx_j = self.mid_interval_indices[j]
                line_length_j = self.line_lengths[interval_idx_j]

                def green_real_param(t, x_pt, src_interval_idx):
                    y_pt = self.param_to_aux_physical(t, src_interval_idx)
                    return np.real(self.green_function(x_pt, y_pt)) * line_length_j

                def green_imag_param(t, x_pt, src_interval_idx):
                    y_pt = self.param_to_aux_physical(t, src_interval_idx)
                    return np.imag(self.green_function(x_pt, y_pt)) * line_length_j

                real_part, _ = quad(green_real_param, t_a, t_b, args=(x_eval, interval_idx_j))
                imag_part, _ = quad(green_imag_param, t_a, t_b, args=(x_eval, interval_idx_j))

                integral_G_dy = real_part + 1j * imag_part
                val_at_x += phi_j * integral_G_dy

            u_scattered[idx_x] = val_at_x

        return u_scattered


def plot_mayavi_surface(x_grid, y_grid, u_tot, bem):
    """Plot the total field magnitude as a 3D surface using Mayavi."""
    s = ml.surf(x_grid.T, y_grid.T, u_tot.T)
    ml.show()


def plot_wave_effects(x_grid, y_grid, u_tot, bem, alpha_rad, k):
    """Plot the incident wave path and total field magnitude using Matplotlib."""
    alpha_deg = np.rad2deg(alpha_rad)
    x_coords = x_grid[0, :]
    y_coords = y_grid[:, 0]

    title_intervals = '; '.join([f"[{s.tolist()}] to [{e.tolist()}]" for s, e in bem.intervals])
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'Wave Fields (k={k}, Lines from {title_intervals}, Angle={alpha_deg}°)', fontsize=14)

    # Subplot 1: Incident Wave Path (Arrows) + Boundaries
    ax1 = plt.subplot(1, 2, 1)
    grid_res = x_grid.shape[0]
    arrow_skip = max(1, grid_res // 10)
    X_q, Y_q = x_grid[::arrow_skip, ::arrow_skip], y_grid[::arrow_skip, ::arrow_skip]

    u_direction = np.cos(alpha_rad)
    v_direction = np.sin(alpha_rad)
    U_q = np.full_like(X_q, u_direction)
    V_q = np.full_like(Y_q, v_direction)

    ax1.quiver(X_q, Y_q, U_q, V_q, angles='xy', scale_units='xy', scale=(k/1.5 or 5),
               color='cyan', headwidth=5, headlength=7, width=0.004, pivot='tail')
    for i, (start, end) in enumerate(bem.intervals):
        ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, label='Boundary $\Gamma$' if i == 0 else "_nolegend_")
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('Incident Wave Path')
    ax1.legend()
    ax1.set_xlim(x_coords.min(), x_coords.max())
    ax1.set_ylim(y_coords.min(), y_coords.max())
    ax1.set_aspect('equal', adjustable='box')

    # Subplot 2: Total Field Magnitude
    ax2 = plt.subplot(1, 2, 2)
    im_total = ax2.imshow(u_tot, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                          origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im_total, ax=ax2, label='$|u_{total}|$')
    for start, end in bem.intervals:
        ax2.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Total Field Magnitude $|u_{inc} + u_{scat}|$')
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def run_bem_test(intervals, alpha_rad, k, n):
    bem = IndirectBEM(intervals=intervals, alpha=alpha_rad, k=k, n=n)

    # Field plots
    grid_res = 40
    # Determine plot bounds based on all line segments
    all_coords = np.vstack([p for interval in bem.intervals for p in interval])
    x_min, x_max = all_coords[:, 0].min() - 2, all_coords[:, 0].max() + 2
    y_min, y_max = all_coords[:, 1].min() - 2, all_coords[:, 1].max() + 2

    x_coords = np.linspace(x_min, x_max, grid_res)
    y_coords = np.linspace(y_min, y_max, grid_res)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    eval_points_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    u_inc_grid = bem.incident_field(eval_points_grid[:, 0], eval_points_grid[:, 1], alpha_rad)

    print(f"Calculating scattered field on a {grid_res}x{grid_res} grid...")
    u_scat_grid = bem.calc_u_scat(eval_points_grid)

    u_tot = np.abs(u_inc_grid + u_scat_grid).reshape(x_grid.shape)

    plot_mayavi_surface(x_grid, y_grid, u_tot, bem)
    plot_wave_effects(x_grid, y_grid, u_tot, bem, alpha_rad, k)

    print("-" * 70 + "\n")
    return bem


if __name__ == '__main__':
    # Change to run with a set of intervals.
    intervals = [([-1, 1], [-0.1, 0.1]), ([0.1, -0.1], [1, -1]),
                 ([1, -1], [2, -1])]
    run_bem_test(intervals, np.pi/4, 10.0, 200)
