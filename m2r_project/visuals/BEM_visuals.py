"""Generate visuals based on the indirect BEM algorithm."""

import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as ml
from ..BEM.indirect_BEM_combined import IndirectBEM


def main():
    """Set up and runs a BEM simulation with predefined parameters."""
    print("Running BEM visuals.")

    geometry_presets = {
        "single_slit_narrow": {
            "intervals": [([-1, 0], [-0.3, 0]), ([0.3, 0], [1, 0])],
            "alpha_rad": np.pi / 2,
            "k": 10.0,
            "n": 250
        },
        "single_slit_wide": {
            "intervals": [([-1, 0], [-0.6, 0]), ([0.6, 0], [1, 0])],
            "alpha_rad": np.pi / 2,
            "k": 10.0,
            "n": 250
        },
        "double_slit": {
            "intervals": [
                ([-1.5, 0], [-0.8, 0]),
                ([-0.4, 0], [0.4, 0]),
                ([0.8, 0], [1.5, 0])
            ],
            "alpha_rad": np.pi / 2,
            "k": 10.0,
            "n": 350
        },
        "corner_reflector": {
            "intervals": [([-1, 0], [0, 0]), ([0, 0], [0, 1])],
            "alpha_rad": np.pi / 4,
            "k": 10.0,
            "n": 250
        },
        "single_plate_horizontal": {
            "intervals": [([-1, 0], [1, 0])],
            "alpha_rad": np.pi / 2,
            "k": 10.0,
            "n": 125
        },
        "single_plate_angled": {
            "intervals": [([-1, -0.5], [1, 0.5])],
            "alpha_rad": np.pi / 2,
            "k": 10.0,
            "n": 125
        },
        "v_shape_scatterer": {
            "intervals": [([-1, 1], [0, 0]), ([0, 0], [1, 1])],
            "alpha_rad": np.pi / 2,
            "k": 10.0,
            "n": 250
        },
        "complex_obstacle": {
            "intervals": [([-1, 1], [-0.1, 0.1]),
                          ([0.1, -0.1], [1, -1]),
                          ([1, -1], [2, -1])],
            "alpha_rad": np.pi / 4,
            "k": 10.0,
            "n": 350
        }
    }

    while True:
        inp = input("Select geometry: \n\
                    1. Complex obstacle \n\
                    2. Corner reflector \n\
                    3. Double-slit \n\
                    4. Single angled plate \n\
                    5. Single horizontal plate \n\
                    6. Narrow single-slit \n\
                    7. Wide single-slit \n\
                    8. V-shape scatterer\n")
        if inp.isdigit() and (int(inp) >= 1 and int(inp) <= 8):
            key_int = int(inp) - 1
            break
        print("\n Option invalid. \n")

    selected_geometry_key = sorted(geometry_presets.keys())[key_int]

    config = geometry_presets[selected_geometry_key]
    print(f"Running BEM for: {selected_geometry_key}")
    run_bem_test(
        intervals=config["intervals"],
        alpha_rad=config["alpha_rad"],
        k=config["k"],
        n=config["n"]
    )


def plot_mayavi_surface(x, y, u_tot, bem):
    """Plot the total field magnitude as a 3D surface using Mayavi."""
    ml.surf(x.T, y.T, u_tot.T)
    ml.show()


def plot_wave_effects(x_grid, y_grid, u_tot, bem, alpha_rad, k):
    """Plot the incident wave path and total field magnitude."""
    alpha_deg = np.rad2deg(alpha_rad)
    x_coords = x_grid[0, :]
    y_coords = y_grid[:, 0]

    title_intervals = '; '.join([f"[{s.tolist()}] to [{e.tolist()}]"
                                 for s, e in bem.intervals])
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'Wave Fields (k={k}, Lines from {title_intervals}, '
                 f'Angle={alpha_deg}°)', fontsize=14)

    # Subplot 1: Incident Wave Path (Arrows) + Boundaries
    ax1 = plt.subplot(1, 2, 1)
    grid_res = x_grid.shape[0]
    arrow_skip = max(1, grid_res // 10)
    X_q, Y_q = x_grid[::arrow_skip, ::arrow_skip], y_grid[::arrow_skip,
                                                          ::arrow_skip]

    u_direction = np.cos(alpha_rad)
    v_direction = np.sin(alpha_rad)
    U_q = np.full_like(X_q, u_direction)
    V_q = np.full_like(Y_q, v_direction)

    ax1.quiver(X_q, Y_q, U_q, V_q, angles='xy', scale_units='xy',
               scale=(k/1.5 or 5), color='cyan', headwidth=5, headlength=7,
               width=0.004, pivot='tail')
    for i, (start, end) in enumerate(bem.intervals):
        ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3,
                 label=r'Boundary $\gamma$' if i == 0 else "_nolegend_")
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title('Incident Wave Path')
    ax1.legend()
    ax1.set_xlim(x_coords.min(), x_coords.max())
    ax1.set_ylim(y_coords.min(), y_coords.max())
    ax1.set_aspect('equal', adjustable='box')

    # Subplot 2: Total Field Magnitude
    ax2 = plt.subplot(1, 2, 2)
    im_total = ax2.imshow(u_tot, extent=[x_coords.min(), x_coords.max(),
                                         y_coords.min(), y_coords.max()],
                          origin='lower', aspect='auto', cmap='viridis',
                          interpolation='nearest')
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
    """Plot the far-field amplitude pattern of the scattered wave."""
    x_vals = np.linspace(0, 2*np.pi, n, endpoint=False)
    y_vals = np.ones(n)
    for i in range(len(y_vals)):
        pos = (r*np.cos(x_vals[i]), r*np.sin(x_vals[i]))

        y_vals[i] = abs(bem.calc_u_scat(pos)[0] * np.sqrt(r) /
                        np.exp(1.0j * bem.k * r))

        if i % 100 == 0:
            print(f"Calculating amplitude sample: {i}/{n} points completed.")

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, linewidth=2)
    ax.set_xlabel("Angle from center θ", fontsize=15)
    ax.set_ylabel("Absolute value of amplitude function |A(θ)|",
                  fontsize=15)

    ticks = np.linspace(0, 2*np.pi, 5)
    xlabels = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(ticks, labels=xlabels)

    plt.show()


def run_bem_test(intervals, alpha_rad, k, n):
    """Run a complete BEM simulation and generate visualization plots."""
    bem = IndirectBEM(intervals=intervals, alpha=alpha_rad, k=k, n=n)

    # Field plots
    grid_res = 40

    # Determine plot bounds based on all line segments
    all_coords = np.vstack([p for interval in bem.intervals
                            for p in interval])
    # Add a 2-unit margin around the object for better visualization
    x_min = max(all_coords[:, 0].min() - 2, -3)
    x_max = min(all_coords[:, 0].max() + 2, 3)

    y_min = max(all_coords[:, 1].min() - 2, -3)
    y_max = min(all_coords[:, 1].max() + 2, 3)

    x_coords = np.linspace(x_min, x_max, grid_res)
    y_coords = np.linspace(y_min, y_max, grid_res)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    eval_points_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    u_inc_grid = bem.incident_field(eval_points_grid[:, 0],
                                    eval_points_grid[:, 1], alpha_rad).real

    print(f"Calculating scattered field on a {grid_res}x{grid_res} grid...")
    u_scat_grid = bem.calc_u_scat(eval_points_grid).real

    u_tot = (u_scat_grid + u_inc_grid).reshape(x_grid.shape)

    plot_mayavi_surface(x_grid, y_grid, u_tot, bem)
    plot_wave_effects(x_grid, y_grid, u_tot, bem, alpha_rad, k)
    # amplitude_sample(bem)

    print("-" * 70 + "\n")
    return bem


if __name__ == '__main__':
    main()
