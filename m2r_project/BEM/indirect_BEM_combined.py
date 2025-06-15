"""Indirect Boundary Element Method for the 2D Helmholtz equation."""
import numpy as np
from math import e, cos, sin
from scipy.integrate import quad
from ..helper.green_function import GreensFunctionCalculator


class IndirectBEM:
    """Combined potential in the Indirect Boundary Element Method."""

    def __init__(self, intervals, alpha, k=10, n=50, use_auxiliary=False):
        """Initialize the BEM solver."""
        self.k = k
        self.alpha = alpha
        self.use_auxiliary = use_auxiliary
        self.intervals = [(np.array(start), np.array(end))
                          for start, end in intervals]

        self.greens_func = GreensFunctionCalculator(self.k)

        # --- Geometric properties of the boundary ---
        self.line_vectors = [end - start for start, end in self.intervals]
        self.line_lengths = [np.linalg.norm(vec) for vec in self.line_vectors]
        self.tangents = [vec / length if length > 0 else np.array([1, 0])
                         for vec, length in zip(
                             self.line_vectors, self.line_lengths)]
        self.normals = [np.array([-tangent[1], tangent[0]])
                        for tangent in self.tangents]

        # --- Discretization setup ---
        total_length = sum(self.line_lengths)
        self.Ns = [int(round(n * length / total_length)) if total_length > 0
                   else int(n / len(self.intervals))
                   for length in self.line_lengths]
        self.N = sum(self.Ns)

        # --- Auxiliary boundary setup ---
        if use_auxiliary:
            self.offset_distances = [0.1 * length / N_i if N_i > 0 else 0
                                     for length, N_i in zip(
                                         self.line_lengths, self.Ns)]
            self.param_to_source_physical = self.param_to_aux_physical
        else:
            self.param_to_source_physical = self.param_to_physical

        # --- BEM Calculation Workflow ---
        print("Generating interval discretisation.")
        self.interval_creator()
        print("Calculating interval midpoints.")
        self.calc_physical_mids()
        self.A = np.zeros((self.N, self.N), dtype=complex)
        self.g_prime = np.zeros(self.N, dtype=complex)
        self.phi = np.zeros(self.N, dtype=complex)
        print(f"Generating matrix A ({self.N}x{self.N}).")
        self.calc_A()
        print("Generating g'.")
        self.calc_g_prime()
        print("Generating field densities.")
        self.calc_phi()

    # ------ Coordinate transformations ------ #

    def param_to_physical(self, t, interval_idx):
        """Convert t in [0, 1] to coordinates on the actual boundary."""
        return (self.intervals[interval_idx][0] +
                t * self.line_vectors[interval_idx])

    def param_to_aux_physical(self, t, interval_idx):
        """Convert t in [0, 1] to coordinates on the auxiliary boundary."""
        physical_point = self.param_to_physical(t, interval_idx)
        offset = (self.offset_distances[interval_idx] *
                  self.normals[interval_idx])
        return physical_point - offset

    # ------ Discretization and Midpoint Calculation ------ #

    def interval_creator(self):
        """Create non-uniform discretisation of intervals (more on edges)."""
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
                n_middle_region = (target_n - n_boundary_region1 -
                                   n_boundary_region3)
            else:
                n_boundary_region1 = target_n // 2
                n_boundary_region3 = target_n - n_boundary_region1
                n_middle_region = 0

            # Define transition points for the graded mesh
            pt_A, pt_D = 0.0, 1.0
            param_scale = (1.0 / (self.k * self.line_lengths[i])
                           if self.line_lengths[i] > 0 else 0.1)
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
                nodes_to_concatenate.append(
                    np.linspace(pt_A, actual_B_endpoint,
                                n_boundary_region1 + 1))
            if (actual_C_startpoint > actual_B_endpoint and
                    n_middle_region > 0):
                nodes_to_concatenate.append(
                    np.linspace(actual_B_endpoint, actual_C_startpoint,
                                n_middle_region + 1))
            if n_boundary_region3 > 0:
                start_point_reg3 = (actual_C_startpoint
                                    if (actual_C_startpoint >
                                        actual_B_endpoint and
                                        n_middle_region > 0)
                                    else actual_B_endpoint)
                nodes_to_concatenate.append(
                    np.linspace(start_point_reg3, pt_D,
                                n_boundary_region3 + 1))

            if nodes_to_concatenate:
                self.all_param_intervals.append(
                    np.unique(np.concatenate(nodes_to_concatenate)))
            else:
                self.all_param_intervals.append(np.array([pt_A, pt_D]))

    def calc_physical_mids(self):
        """Calculate midpoints (collocation points) of each element."""
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
        u_inc = np.array([self.incident_field(mid[0], mid[1], self.alpha)
                          for mid in self.mids])
        duinc_dx1 = 1j * self.k * cos(self.alpha) * u_inc
        duinc_dx2 = 1j * self.k * sin(self.alpha) * u_inc
        normals_array = np.array([self.normals[i]
                                  for i in self.mid_interval_indices])
        self.g_prime = -(normals_array[:, 0] * duinc_dx1 +
                         normals_array[:, 1] * duinc_dx2)

    def _kernel_integrand(self, t, x_coll, n_x, n_y, src_idx):
        """Compute the complex kernel integrand value."""
        y_source = self.param_to_source_physical(t, src_idx)
        return (-self.k**2 *
                self.greens_func.greens_func(x_coll, y_source) +
                1j * self.greens_func.dir_deriv(x_coll, y_source, n_x) -
                self.greens_func.mixed_dir_deriv(x_coll, y_source, n_x, n_y)
                )

    def kernel_real_param(self, t, x_coll, n_x, n_y, src_idx):
        """Compute real part of kernel integrand in parametric form."""
        jacobian = self.line_lengths[src_idx]
        integ_val = self._kernel_integrand(t, x_coll, n_x, n_y, src_idx)
        return np.real(integ_val) * jacobian

    def kernel_imag_param(self, t, x_coll, n_x, n_y, src_idx):
        """Compute imaginary part of kernel integrand in parametric form."""
        jacobian = self.line_lengths[src_idx]
        integ_val = self._kernel_integrand(t, x_coll, n_x, n_y, src_idx)
        return np.imag(integ_val) * jacobian

    def calc_A(self):
        """Calculate the influence matrix A."""
        for i in range(self.N):
            x_mid_point = self.mids[i]
            normal_at_x_mid = self.normals[self.mid_interval_indices[i]]

            for j in range(self.N):
                t_a_j, t_b_j = self.element_param_bounds[j]
                interval_idx_j = self.mid_interval_indices[j]
                normal_at_y_mid = self.normals[interval_idx_j]

                # --- TOGGLEABLE LOGIC ---
                # no auxiliary => calculating diagonal elements analytically
                # auxiliary => standard numerical quadrature.
                if not self.use_auxiliary and i == j:
                    bound = self.element_param_bounds[i]
                    L = self.line_lengths[self.mid_interval_indices[i]]
                    L_i = L * abs(bound[0] - bound[1])
                    diag_val = (-L * L_i * 1.0j * self.k ** 2 *
                                (np.pi + 2.0j * (np.log(L / 2) +
                                 np.log(self.k * L_i / 4) +
                                 np.euler_gamma - 1)) / (8 * np.pi))
                    self.A[i, j] = diag_val
                else:
                    real_part, _ = quad(self.kernel_real_param, t_a_j, t_b_j,
                                        args=(x_mid_point, normal_at_x_mid,
                                              normal_at_y_mid, interval_idx_j))
                    imag_part, _ = quad(self.kernel_imag_param, t_a_j, t_b_j,
                                        args=(x_mid_point, normal_at_x_mid,
                                              normal_at_y_mid, interval_idx_j))
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

    def _scattered_field_integrand(self, x_pt, y_pt, normal_at_y, phi_j):
        """Compute the complex integrand for the scattered fieldA."""
        return (
            (self.greens_func.dir_deriv(x_pt, y_pt, normal_at_y) -
             1j * self.k * self.greens_func.greens_func(x_pt, y_pt)) * phi_j
        )

    def green_real_param(self, t, x_pt, src_interval_idx, normal_at_y,
                         phi_j, line_length_j):
        """Compute real part of green integrand in parametric form."""
        y_pt = self.param_to_source_physical(t, src_interval_idx)
        integrand_val = self._scattered_field_integrand(
            x_pt, y_pt, normal_at_y, phi_j)
        return np.real(integrand_val) * line_length_j

    def green_imag_param(self, t, x_pt, src_interval_idx, normal_at_y,
                         phi_j, line_length_j):
        """Compute imaginary part of green integrand in parametric form."""
        y_pt = self.param_to_source_physical(t, src_interval_idx)
        integrand_val = self._scattered_field_integrand(
            x_pt, y_pt, normal_at_y, phi_j)
        return np.imag(integrand_val) * line_length_j

    def calc_u_scat(self, x_points):
        """Calculate the scattered field at a set of evaluation points."""
        u_scattered = np.zeros(len(x_points), dtype=complex)

        for idx_x, x_eval in enumerate(x_points):
            if idx_x > 0 and idx_x % 100 == 0:
                print(f"Calculating scattered field: {idx_x}/"
                      f"{len(x_points)} points completed.")

            val_at_x = 0.0 + 0.0j
            for j in range(self.N):
                phi_j = self.phi[j]
                t_a, t_b = self.element_param_bounds[j]
                interval_idx_j = self.mid_interval_indices[j]
                line_length_j = self.line_lengths[interval_idx_j]
                normal_at_y = self.normals[interval_idx_j]

                real_part, _ = quad(
                    self.green_real_param, t_a, t_b,
                    args=(x_eval, interval_idx_j, normal_at_y, phi_j,
                          line_length_j)
                )
                imag_part, _ = quad(
                    self.green_imag_param, t_a, t_b,
                    args=(x_eval, interval_idx_j, normal_at_y, phi_j,
                          line_length_j)
                )
                val_at_x += (real_part + 1j * imag_part)

            u_scattered[idx_x] = val_at_x

        return u_scattered
