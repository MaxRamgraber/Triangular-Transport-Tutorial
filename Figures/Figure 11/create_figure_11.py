import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.special

plt.close("all")

plt.figure(figsize=(12,8))

plt.subplot(2,1,1)

# Create evaluation points
x       = np.linspace(-7,7,1001)

# Evaluate the order 1 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,1])(x)*np.exp(-x**2/4)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:cerulean', 
    label   = '$\mathcal{H}_{1}$ (order 1)')

# Evaluate the order 2 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,0,1])(x)*np.exp(-x**2/4)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:grass green', 
    label   = '$\mathcal{H}_{2}$ (order 2)')

# Evaluate the order 3 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,0,0,1])(x)*np.exp(-x**2/4)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:goldenrod', 
    label   = '$\mathcal{H}_{3}$ (order 3)')


# Evaluate the order 4 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,0,0,0,1])(x)*np.exp(-x**2/4)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:orangish red', 
    label   = '$\mathcal{H}_{4}$ (order 4)')

plt.xlabel('coordinates $x$')
plt.ylabel('Hermite function $\mathcal{H}_{n}$ (normalized)')
plt.legend(frameon=False,ncol = 1)

# Remove all axis ticks
plt.tick_params(left=False,
                labelleft=False)

plt.title("$\mathbf{A}$: Hermite functions", loc="left")

plt.subplot(2,1,2)


edge_control_radius = 3

# Create evaluation points
x       = np.linspace(-4,4,1001)

# Evaluate the order 1 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,1])(x)*(2*np.minimum(1,np.abs(x/edge_control_radius))**3 - 3*np.minimum(1,np.abs(x/edge_control_radius))**2 + 1)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:cerulean', 
    label   = '$\mathcal{H}_{1}^{EC}$ (order 1)')

# Evaluate the order 2 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,0,1])(x)*(2*np.minimum(1,np.abs(x/edge_control_radius))**3 - 3*np.minimum(1,np.abs(x/edge_control_radius))**2 + 1)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:grass green', 
    label   = '$\mathcal{H}_{2}^{EC}$ (order 2)')

# Evaluate the order 3 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,0,0,1])(x)*(2*np.minimum(1,np.abs(x/edge_control_radius))**3 - 3*np.minimum(1,np.abs(x/edge_control_radius))**2 + 1)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:goldenrod', 
    label   = '$\mathcal{H}_{3}^{EC}$ (order 3)')


# Evaluate the order 4 Hermite function
y       = np.polynomial.hermite_e.HermiteE([0,0,0,0,1])(x)*(2*np.minimum(1,np.abs(x/edge_control_radius))**3 - 3*np.minimum(1,np.abs(x/edge_control_radius))**2 + 1)
y       = y/np.max(y)   # Normalize
plt.plot(
    x,
    y,
    color   = 'xkcd:orangish red', 
    label   = '$\mathcal{H}_{4}^{EC}$ (order 4)')


plt.xlabel('coordinates $x$')
plt.ylabel('edge-controlled Hermite $\mathcal{H}_{n}^{EC}$ (normalized)')
plt.legend(frameon=False,ncol = 1)
plt.title("$\mathbf{B}$: edge-controlled Hermite polynomials", loc="left")

# Remove all axis ticks
plt.tick_params(left=False,
                labelleft=False)

plt.savefig('edge_controlled_hermite.png',dpi=600,bbox_inches='tight')
plt.savefig('edge_controlled_hermite.pdf',dpi=600,bbox_inches='tight')