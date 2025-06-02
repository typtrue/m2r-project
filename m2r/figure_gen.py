from scipy.special import hankel1
import mayavi.mlab as ml
import numpy as np

def G(r, r0, k):
    """Green function for 2D Helmholtz."""
    return 1.0j * hankel1(0, k * np.linalg.norm(r - r0)) / 4

def DG(r, r_0, k):
    """Partial derivative of Green function wrt. y in second variable."""
    w, x = r_0
    y, z = r

    return 1.0j * k * (z - x) * hankel1(1, k * np.linalg.norm(r - r_0)) / (4 * np.linalg.norm(r - r_0))

n = 1000
k = 10


xvals = np.linspace(-3, 3, n)
yvals = np.linspace(-3, 3, n)

x, y = np.meshgrid(xvals, yvals)

stack = np.column_stack((x.flatten(), y.flatten()))

z_vals = np.ones(len(stack))

r0 = np.array([0, 0])

for i in range(len(z_vals)):
    z_vals[i] = G(stack[i], r0, k).real
    if abs(z_vals[i]) >= 5:
        z_vals[i] = None
    if i % 1000 == 0:
        print(f"{i}/{n**2}")

z = np.reshape(z_vals, (-1, n)) 

s = ml.surf(x.T, y.T, z.T)

ml.show()
