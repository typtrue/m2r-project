from scipy.special import hankel1
import mayavi.mlab as ml
import numpy as np
import matplotlib.pyplot as plt

def G(r, r0, k):
    """Green function for 2D Helmholtz."""
    return 1.0j * hankel1(0, k * np.linalg.norm(r - r0)) / 4

def DG(self, r, r_0):
    """Partial derivative of Green function wrt. y in second variable."""
    w, x = r
    y, z = r_0

    return 1.0j * self.k * (x - z) * hankel1(1, self.k * np.linalg.norm(r - r_0)) / (4 * np.linalg.norm(r - r_0))


fig, ax = plt.subplots()
ax.set_xlabel(r"$\tilde{x}$", fontsize=10)
ax.set_ylabel(r"$\tilde{y}$", fontsize=10)
ax.plot([-3, 0], [0, 0], 'r-', linewidth=3, label='Boundary $\gamma$')
ax.plot([0, 0], [-3, 3], '--', color="black", linewidth=2)
ax.set_xbound(-3, 3)
ax.set_ybound(-3, 3)
ax.set_box_aspect(1)

ax.text(-1.5, -1.5, '1', fontsize=20)
ax.text(-1.5, 1.5, '2', fontsize=20)
ax.text(1.5, -0.1, '3', fontsize=20)

ax.legend()

plt.show()

# n = 1000
# k = 10


# xvals = np.linspace(-3, 3, n)
# yvals = np.linspace(-3, 3, n)

# x, y = np.meshgrid(xvals, yvals)

# stack = np.column_stack((x.flatten(), y.flatten()))

# z_vals = np.ones(len(stack))

# r0 = np.array([0, 0])

# for i in range(len(z_vals)):
#     z_vals[i] = G(stack[i], r0, k).real
#     if abs(z_vals[i]) >= 5:
#         z_vals[i] = None
#     if i % 1000 == 0:
#         print(f"{i}/{n**2}")

# z = np.reshape(z_vals, (-1, n)) 

# s = ml.surf(x.T, y.T, z.T)

# ml.show()
