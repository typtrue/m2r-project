import numpy as np
from scipy.special import hankel1
from math import e, cos, sin
from scipy.integrate import quad
import matplotlib.pyplot as plt

class IndirectBEM:
    """
    Implements the Indirect Boundary Element Method for the 2D Helmholtz equation.

    This class can solve the acoustic scattering problem using two methods for
    handling the singular integrals:
    1.  Standard Method (Default): Analytically calculates the integral for singular
        diagonal elements and uses numerical quadrature for others.
    2.  Auxiliary Method: Shifts the integration boundary slightly to avoid
        singularities altogether, allowing numerical quadrature for all elements.
    """
    def __init__(self, intervals, alpha, k=10, n=100, use_auxiliary=False):
        """
        Initializes the BEM solver.

        Args:
            intervals (list of tuples): A list of start and end points for each line segment of the boundary.
            alpha (float): The angle of incidence of the plane wave in radians.
            k (float, optional): The wave number. Defaults to 10.
            n (int, optional): The total number of discretization points. Defaults to 100.
            use_auxiliary (bool, optional): If True, uses the auxiliary boundary method to handle singularities.
                                            If False (default), uses analytical integration for singular elements.
        """
        self.k = k
        self.alpha = alpha
        self.use_auxiliary = use_auxiliary
        self.intervals = [(np.array(start), np.array(end)) for start, end in intervals]

        # --- Geometric properties of the boundary ---
        self.line_vectors = [end - start for start, end in self.intervals]
        self.line_lengths = [np.linalg.norm(vec) for vec in self.line_vectors]
        self.tangents = [vec / length if length > 0 else np.array([1, 0]) for vec, length in zip(self.line_vectors, self.line_lengths)]
        self.normals = [np.array([-tangent[1], tangent[0]]) for tangent in self.tangents]

        # --- Discretization setup ---
        total_length = sum(self.line_lengths)
        self.Ns = [int(round(n * length / total_length)) if total_length > 0 else int(n / len(self.intervals)) for length in self.line_lengths]
        self.N = sum(self.Ns)
        
        # --- Auxiliary boundary setup ---
        # The offset distance delta is calculated, but only used if use_auxiliary is True.
        self.offset_distances = [0.1 * length / N_i if N_i > 0 else 0 for length, N_i in zip(self.line_lengths, self.Ns)]

        # --- BEM Calculation Workflow ---
        self.interval_creator()
        self.calc_physical_mids()
        self.A = np.zeros((self.N, self.N), dtype=complex)
        self.g_prime = np.zeros(self.N, dtype=complex)
        self.phi = np.zeros(self.N, dtype=complex)
        self.calc_A()
        self.calc_g_prime()
        self.calc_phi()

    # ------ Green's functions and their derivatives ------ #

    def G(self, x, y):
        """Green's function for 2D Helmholtz."""
        diff = np.array(x) - np.array(y)
        R = np.linalg.norm(diff)
        if np.isclose(R, 0):
            return np.inf + 0j # Should be handled by analytical integration or auxiliary method
        return 1j/4 * hankel1(0, self.k * R)

    def DG(self, x, y, normal_vec):
        """Normal derivative of the Green's function with respect to the source normal."""
        diff = x - y
        R = np.linalg.norm(diff)
        if np.isclose(R, 0):
            return np.inf + 0j
        direction = diff / R
        normal_deriv = np.dot(direction, normal_vec)
        return normal_deriv * (1j * self.k / 4.0) * hankel1(1, self.k * R)

    def D2G(self, x, y, normal_vec_x, normal_vec_y):
        """Second partial derivative of the Green's function."""
        diff = x - y
        R = np.linalg.norm(diff)
        if np.isclose(R, 0):
            return np.inf + 0j
        s = hankel1(0, self.k * R) - hankel1(2, self.k * R)
        return (1.0j * self.k ** 2 * np.dot(diff, normal_vec_x) * np.dot(diff, normal_vec_y) * s) / (8 * R ** 2)

    # ------ Coordinate transformations ------ #

    def param_to_physical(self, t, interval_idx):
        """Convert parameter t in [0, 1] to physical coordinates on the actual boundary."""
        return self.intervals[interval_idx][0] + t * self.line_vectors[interval_idx]

    def param_to_aux_physical(self, t, interval_idx):
        """Convert parameter t in [0, 1] to physical coordinates on the auxiliary boundary."""
        physical_point = self.param_to_physical(t, interval_idx)
        offset = self.offset_distances[interval_idx] * self.normals[interval_idx]
        return physical_point - offset

    # ------ Discretization and Midpoint Calculation ------ #

    def interval_creator(self):
        """Create graded parameter intervals to have more points near corners."""
        self.all_param_intervals = []
        for i in range(len(self.intervals)):
            if self.Ns[i] == 0:
                self.all_param_intervals.append([])
                continue

            target_n = self.Ns[i]
            # Refine mesh near endpoints based on wave number k
            n_boundary = int(round(5 * self.k))
            if target_n >= 2 * n_boundary:
                n_boundary_region1 = n_boundary
                n_boundary_region3 = n_boundary
                n_middle_region = target_n - n_boundary_region1 - n_boundary_region3
            else:
                n_boundary_region1 = target_n // 2
                n_boundary_region3 = target_n - n_boundary_region1
                n_middle_region = 0

            # Define transition points for the graded mesh
            pt_A, pt_D = 0.0, 1.0
            param_scale = 1.0 / (self.k * self.line_lengths[i]) if self.line_lengths[i] > 0 else 0.1
            ideal_transition_B = pt_A + param_scale
            ideal_transition_C = pt_D - param_scale
            
            # Ensure transition points are within the [0, 1] interval
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
                start_point_reg3 = actual_C_startpoint if (actual_C_startpoint > actual_B_endpoint and n_middle_region > 0) else actual_B_endpoint
                nodes_to_concatenate.append(np.linspace(start_point_reg3, pt_D, n_boundary_region3 + 1))
            
            if nodes_to_concatenate:
                self.all_param_intervals.append(np.unique(np.concatenate(nodes_to_concatenate)))
            else:
                 self.all_param_intervals.append(np.array([pt_A, pt_D]))

    def calc_physical_mids(self):
        """Calculate physical midpoints (collocation points) of each element."""
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

    # ------ BEM Core Calculations ------ #

    def incident_field(self, x1, x2, alpha):
        """Calculate the incident plane wave."""
        return e**(1j * self.k * (x1 * cos(alpha) + x2 * sin(alpha)))

    def calc_g_prime(self):
        """Calculate the boundary condition g' = -∂u_inc/∂n."""
        u_inc = np.array([self.incident_field(mid[0], mid[1], self.alpha) for mid in self.mids])
        duinc_dx1 = 1j * self.k * cos(self.alpha) * u_inc
        duinc_dx2 = 1j * self.k * sin(self.alpha) * u_inc
        normals_array = np.array([self.normals[i] for i in self.mid_interval_indices])
        self.g_prime = -(normals_array[:, 0] * duinc_dx1 + normals_array[:, 1] * duinc_dx2)

    def calc_A(self):
        """
        Calculate the influence matrix A.
        This method uses the `self.use_auxiliary` flag to decide how to handle
        the singular integrals on the diagonal elements of the matrix.
        """
        k = self.k
        # Determine the source point mapping based on the chosen method.
        # The source point (y) is on the auxiliary boundary if use_auxiliary is True.
        param_to_source_physical = self.param_to_aux_physical if self.use_auxiliary else self.param_to_physical

        for i in range(self.N):
            # The collocation point (x) is always on the original boundary.
            x_mid_point = self.mids[i]
            normal_at_x_mid = self.normals[self.mid_interval_indices[i]]

            for j in range(self.N):
                t_a_j, t_b_j = self.element_param_bounds[j]
                interval_idx_j = self.mid_interval_indices[j]
                normal_at_y_mid = self.normals[interval_idx_j]

                # --- TOGGLEABLE LOGIC ---
                # If not using the auxiliary method, we must handle the singularity
                # on the diagonal elements (i==j) analytically.
                if not self.use_auxiliary and i == j:
                    bound = self.element_param_bounds[i]
                    L = self.line_lengths[self.mid_interval_indices[i]]
                    L_i = L * abs(bound[0] - bound[1])
                    # Analytical solution for the singular integral
                    diag_val = -L * L_i * 1.0j * self.k ** 2 * (np.pi + 2.0j * (np.log(L / 2) + np.log(self.k * L_i / 4) + np.euler_gamma - 1)) / (8 * np.pi)
                    self.A[i, j] = diag_val
                else:
                    # If using the auxiliary method, or for off-diagonal elements,
                    # we can use standard numerical quadrature.
                    def kernel_real_param(t, x_coll, n_x, n_y, src_idx):
                        y_source = param_to_source_physical(t, src_idx)
                        jacobian = self.line_lengths[src_idx]
                        integrand = -k**2 * self.G(x_coll, y_source) + 1j * self.DG(x_coll, y_source, n_x) - self.D2G(x_coll, y_source, n_x, n_y)
                        return np.real(integrand) * jacobian

                    def kernel_imag_param(t, x_coll, n_x, n_y, src_idx):
                        y_source = param_to_source_physical(t, src_idx)
                        jacobian = self.line_lengths[src_idx]
                        integrand = -k**2 * self.G(x_coll, y_source) + 1j * self.DG(x_coll, y_source, n_x) - self.D2G(x_coll, y_source, n_x, n_y)
                        return np.imag(integrand) * jacobian

                    real_part, _ = quad(kernel_real_param, t_a_j, t_b_j, args=(x_mid_point, normal_at_x_mid, normal_at_y_mid, interval_idx_j))
                    imag_part, _ = quad(kernel_imag_param, t_a_j, t_b_j, args=(x_mid_point, normal_at_x_mid, normal_at_y_mid, interval_idx_j))
                    self.A[i, j] = real_part + 1j * imag_part

    def calc_phi(self):
        """Solve the linear system to find the density function phi."""
        try:
            # For Neumann problem, the BIE is (i/2)phi - A*phi = g'
            # (Note: This is based on eq 3.22, 3.24 from the document)
            B = (1.0j * np.identity(self.N) / 2.0) - self.A
            self.phi = np.linalg.solve(B, self.g_prime)
        except np.linalg.LinAlgError as e:
            print(f"Error solving linear system: {e}")
            self.phi = None

    def calc_u_scat(self, x_points):
        """
        Calculate the scattered field at a set of evaluation points.
        This method also uses the `self.use_auxiliary` flag to determine the
        integration path for calculating the scattered field.
        """
        u_scattered = np.zeros(len(x_points), dtype=complex)
        
        # Determine the integration path (source points y)
        param_to_source_physical = self.param_to_aux_physical if self.use_auxiliary else self.param_to_physical

        for idx_x, x_eval in enumerate(x_points):
            if idx_x > 0 and idx_x % 100 == 0:
                print(f"Calculating scattered field: {idx_x}/{len(x_points)} points completed.")
            
            val_at_x = 0.0 + 0.0j
            for j in range(self.N):
                phi_j = self.phi[j]
                t_a, t_b = self.element_param_bounds[j]
                interval_idx_j = self.mid_interval_indices[j]
                line_length_j = self.line_lengths[interval_idx_j]
                normal_at_y = self.normals[interval_idx_j]

                def green_real_param(t, x_pt, src_interval_idx):
                    y_pt = param_to_source_physical(t, src_interval_idx)
                    integrand = (self.DG(x_pt, y_pt, normal_at_y) - 1j * self.k * self.G(x_pt, y_pt)) * phi_j
                    return np.real(integrand) * line_length_j

                def green_imag_param(t, x_pt, src_interval_idx):
                    y_pt = param_to_source_physical(t, src_interval_idx)
                    integrand = (self.DG(x_pt, y_pt, normal_at_y) - 1j * self.k * self.G(x_pt, y_pt)) * phi_j
                    return np.imag(integrand) * line_length_j

                real_part, _ = quad(green_real_param, t_a, t_b, args=(x_eval, interval_idx_j))
                imag_part, _ = quad(green_imag_param, t_a, t_b, args=(x_eval, interval_idx_j))
                val_at_x += (real_part + 1j * imag_part)

            u_scattered[idx_x] = val_at_x

        return u_scattered


def plot_mayavi_surface(x, y, u_tot, bem):
    """Plot the total field magnitude as a 3D surface using Mayavi."""
    s = ml.surf(x.T, y.T, u_tot.T)
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
        ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, label='Boundary $\gamma$' if i == 0 else "_nolegend_")
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
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
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_title('Real part of total field $Re(u_{inc} + u_{scat})$')
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def amplitude_sample(bem, n=1000, r=10**7):
    x_vals = np.linspace(0, 2*np.pi, n, endpoint=False)
    y_vals = np.ones(n)
    for i in range(len(y_vals)):
        pos = (r*np.cos(x_vals[i]), r*np.sin(x_vals[i]))

        y_vals[i] = abs(bem.calc_u_scat(pos)[0] * np.sqrt(r) / np.exp(1.0j * bem.k * r))

        if i % 100 == 0:
            print(f"{i}/{n}")

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, linewidth=2)
    ax.set_xlabel("Angle from center θ", fontsize=15)
    ax.set_ylabel("Absolute value of amplitude function |A(θ)|", fontsize=15)

    ticks = np.linspace(0, 2*np.pi, 5)
    xlabels = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(ticks, labels=xlabels)

    plt.show()


def run_bem_test(intervals, alpha_rad, k, n):
    bem = IndirectBEM(intervals=intervals, alpha=alpha_rad, k=k, n=n)

    # Field plots
    grid_res = 40


    # Determine plot bounds based on all line segments
    all_coords = np.vstack([p for interval in bem.intervals for p in interval])
    # x_min, x_max = all_coords[:, 0].min() - 2, all_coords[:, 0].max() + 2
    # y_min, y_max = all_coords[:, 1].min() - 2, all_coords[:, 1].max() + 2

    x_min, x_max = -3, 3
    y_min, y_max = -3, 3

    x_coords = np.linspace(x_min, x_max, grid_res)
    y_coords = np.linspace(y_min, y_max, grid_res)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    eval_points_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    u_inc_grid = bem.incident_field(eval_points_grid[:, 0], eval_points_grid[:, 1], alpha_rad).real

    print(f"Calculating scattered field on a {grid_res}x{grid_res} grid...")
    u_scat_grid = bem.calc_u_scat(eval_points_grid).real

    u_tot = (u_scat_grid + u_inc_grid).reshape(x_grid.shape)

    # plot_mayavi_surface(x_grid, y_grid, u_tot, bem)
    plot_wave_effects(x_grid, y_grid, u_tot, bem, alpha_rad, k)
    # amplitude_sample(bem)

    print("-" * 70 + "\n")
    return bem


if __name__ == '__main__':
    # Change to run with a set of intervals.
    # intervals = [([-1, 1], [-0.1, 0.1]), ([0.1, -0.1], [1, -1]),
    #              ([1, -1], [2, -1])]
    # run_bem_test(intervals, np.pi/4, 10.0, 200)

    intervals = [([-1, 0], [-0.3, 0]), ([0.3, 0], [1, 0])]
    run_bem_test(intervals, np.pi/2, 10.0, 400)
