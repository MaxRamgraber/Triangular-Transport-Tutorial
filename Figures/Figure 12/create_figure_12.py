import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.special

plt.close("all")

plt.figure(figsize=(12,4))

# Create evaluation points
Delta   = np.linspace(-3,3,1001)

# Evaluate a radial basis function
rbf     = 1/np.sqrt(2*np.pi)*np.exp(-Delta**2)
rbf     = rbf/np.max(rbf)   # Normalize
plt.plot(
    Delta,
    rbf,
    color   = 'xkcd:cerulean', 
    label   = 'radial basis function')

# Evaluate the integrated radial basis function
irbf    = 1/2*(1 + scipy.special.erf(Delta))
plt.plot(
    Delta,
    irbf,
    color   = 'xkcd:grass green', 
    label   = 'integrated radial basis function')

# Evaluate the integrated radial basis function
let     = 1/2*(Delta*(1 - scipy.special.erf(Delta)) - np.sqrt(2/np.pi)*np.exp(-Delta**2))
let     = let/np.abs(np.min(let))  # Normalize
plt.plot(
    Delta,
    let,
    color   = 'xkcd:goldenrod', 
    label   = 'left edge term')


# Evaluate the integrated radial basis function
ret     = 1/2*(Delta*(1 + scipy.special.erf(Delta)) + np.sqrt(2/np.pi)*np.exp(-Delta**2))
ret     = ret/np.max(ret)   # Normalize
plt.plot(
    Delta,
    ret,
    color   = 'xkcd:orangish red', 
    label   = 'right edge term')

plt.xlabel('local coordinates $x_{k}^{loc,i}$')
plt.ylabel('function output (normalized)')
plt.legend(frameon=False,ncol = 1)

# Remove all axis ticks
plt.tick_params(left=False,
                labelleft=False)

plt.savefig('radial_basis_functions.png',dpi=600,bbox_inches='tight')
plt.savefig('radial_basis_functions.pdf',dpi=600,bbox_inches='tight')