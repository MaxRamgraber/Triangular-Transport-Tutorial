import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools
import sklearn.neighbors
import math
import pickle
import matplotlib
from matplotlib import colors
import os
from matplotlib import gridspec
from transport_map import *

root_directory = os.path.dirname(os.path.realpath(__file__))

np.random.seed(0)

plt.close('all')

def sample_multimodal_distribution(size):
    
    randseeds   = np.random.uniform(size=size)
    
    seeds1  = np.where(np.logical_and(
        randseeds >= 0,
        randseeds < 1/3))
    
    seeds2  = np.where(np.logical_and(
        randseeds >= 1/3,
        randseeds < 2/3))
    
    seeds3  = np.where(randseeds >= 2/3)
    
    X   = np.zeros((size,2))
    
    X[seeds1,:] = scipy.stats.multivariate_normal.rvs(
        mean    = [-1.5,-1.5],
        cov     = np.identity(2)*0.05,
        size    = len(seeds1[0]))
    
    X[seeds2,:] = scipy.stats.multivariate_normal.rvs(
        mean    = [-1.5,1.5],
        cov     = np.identity(2)*0.05,
        size    = len(seeds2[0]))
    
    X[seeds3,:] = scipy.stats.multivariate_normal.rvs(
        mean    = [1.5,1.5],
        cov     = np.identity(2)*0.05,
        size    = len(seeds3[0]))
    
    return X
    

# Training ensemble size
N   = 10000

# Draw that many samples
X   = sample_multimodal_distribution(N)

# =============================================================================
# Train the transport map
# =============================================================================

# Let's see if tm can map standard Gaussian samples to the target
norm_samples    = scipy.stats.norm.rvs(size=(3000,2))

# Get the directional color values for the Gaussian samples
color_values_directional = []
for i in range(norm_samples.shape[0]):
    
    # Convert their location relative to the origin to an angle
    dummy   = math.atan2(norm_samples[i,1],norm_samples[i,0])
    
    # Then extract the corresponding colour on a circular colormap (hsv)
    dummy   = list(matplotlib.cm.hsv((dummy+np.pi)/(2*np.pi)))
    
    # And write them into the color list
    color_values_directional.append(dummy)
    
# Convert the list into an array
color_values_directional    = np.asarray(color_values_directional)

plt.close('all')
plt.figure(figsize=(12,12)) # 14,21

gs  = matplotlib.gridspec.GridSpec(
    nrows           = 3,
    ncols           = 3,
    wspace          = 0.1, 
    hspace          = 0.1)

# Learn a Kernel density estimate of this distribution
kde = sklearn.neighbors.KernelDensity(bandwidth=0.1)
kde.fit(X)

# Create a slightly different meshgrid
x,y = np.meshgrid(
    np.linspace(-3,3,101),
    np.linspace(-3,3,101))

# Extract the target distribution's density from the Kernel Density Estimator
xy  = np.column_stack((
    np.ndarray.flatten(x),
    np.ndarray.flatten(y)))
z   = np.exp(kde.score_samples(xy))
z   = z.reshape((101,101))

resolution  = 31
binpos      = np.linspace(-3,3,resolution+1)[1:]
binpos      -= (binpos[1]-binpos[0])/2
binwidth    = binpos[1]-binpos[0]
bins        = [[x-binwidth/2,x+binwidth/2] for x in binpos]

def triang(N):
    
    x = 0
    for i in np.arange(1,N+1,1):
        x += i
        
    return x


for complexity in [0,1,2]:
    
    for mi,maxorder in enumerate([1,3,10]):
        
        # ---------------------------------------------------------------------
        # Find out how complex the crossterm expression is
        # Use this as a reference to determine the parameterization of the other
        # variables
        # ---------------------------------------------------------------------
        
        # Nonmonotone dim 1
        varnum = 1

        # Monotone dim 1
        varnum += maxorder
        
        # Nonmonotone dim 2
        varnum += 1 + maxorder      
        
        # Monotone dim 2
        varnum += triang(maxorder)


        if complexity == 0:  
            
            # -----------------------------------------------------------------
            
            if maxorder > 1:
            
                # Find the dynamical order to match complexity

                order       = 1
                varnum_dyn  = []
                repeat      = True
                
                while repeat:
                    
                    varnum_dyn.append(2 + 2*order)
                    
                    if varnum_dyn[-1] > varnum:
                        
                        repeat = False
                        
                    else:
                        
                        order   += 1
                
                # Determine the best maxorder
                if varnum_dyn[-1] - varnum <= varnum - varnum_dyn[-2] or order <= 1:
                    
                    # A bit more complexity is closer to the target coefficient count
                    order     = order
                    
                else:
                    
                    # A bit less complexity is closer to the target coefficient count
                    order     -= 1
                    
            else:
                
                order   = 1
                
            # -----------------------------------------------------------------
            
            # Create empty lists for the map component specifications
            monotone    = []
            nonmonotone = []
            
            # Specify the map components
            for k in range(2):
                
                # Level 1: One list entry for each dimension (two in total)
                monotone.append([])
                nonmonotone.append([[]]) # An empty list "[]" denotes a constant
            
                # Go through each polynomial order
                for o in range(order):
                    
                    # Monotone part -------------------------------------------------------
                    
                    # Level 2: Specify the polynomial order of this term;
                    # It's a Hermite function
                    if o == 0:
                        monotone[-1].append([])
                    else:
                        monotone[-1].append(
                            [k]*(o+1)+['HF'])
            
        elif complexity == 1:      
            
            # -----------------------------------------------------------------
            
            if maxorder > 1:
    
                
                order       = 1
                varnum_dyn  = []
                repeat      = True
                
                while repeat:
                    
                    varnum_dyn.append(2 + 3*order)
                    
                    if varnum_dyn[-1] > varnum:
                        
                        repeat = False
                        
                    else:
                        
                        order   += 1
                
                # Determine the best maxorder
                if varnum_dyn[-1] - varnum <= varnum - varnum_dyn[-2] or order <= 1:
                    
                    # A bit more complexity is closer to the target coefficient count
                    order     = order
                    
                else:
                    
                    # A bit less complexity is closer to the target coefficient count
                    order     -= 1
                    
            else:
                
                order   = 1
                
            # -----------------------------------------------------------------

            # Create empty lists for the map component specifications
            monotone    = []
            nonmonotone = []
            
            # Specify the map components
            for k in range(2):
                
                # Level 1: One list entry for each dimension (two in total)
                monotone.append([])
                nonmonotone.append([[]]) # An empty list "[]" denotes a constant
            
                # Go through each polynomial order
                for o in range(order):
                    
                    # Nonmonotone part ----------------------------------------------------
                    
                    if k > 0:
                        # Level 2: Specify the polynomial order of this term;
                        # It's a Hermite function term
                        if o == 0:
                            nonmonotone[-1].append([k-1]*(o+1))
                        else:
                            nonmonotone[-1].append([k-1]*(o+1)+['HF'])
                    
                    # Monotone part -------------------------------------------------------
                    
                    # Level 2: Specify the polynomial order of this term;
                    if o == 0:
                        monotone[-1].append([])
                    else:
                        monotone[-1].append(
                            [k]*(o+1)+['HF'])
                    
                    # # It's a Hermite function
                    # monotone[-1].append(
                    #     [k]*(o+1)+['HF'])
        
        elif complexity == 2:      
            
            order   = maxorder
            
            # Create empty lists for the map component specifications
            monotone    = []
            nonmonotone = []
            
            # Specify the map components
            for k in range(2):
                
                # Level 1: One list entry for each dimension (two in total)
                monotone.append([])
                nonmonotone.append([[]]) # An empty list "[]" denotes a constant
            
                # Go through each polynomial order
                for o in range(order):
                    
                    # Nonmonotone part ----------------------------------------------------
                    
                    if k > 0:
                        # Level 2: Specify the polynomial order of this term;
                        # It's a Hermite function term
                        if o == 0:
                            nonmonotone[-1].append([k-1]*(o+1))
                        else:
                            nonmonotone[-1].append([k-1]*(o+1)+['HF'])
                    
                    # Monotone part -------------------------------------------------------
                    
                    # For the monotone part, we consider cross-terms. This list creates all
                    # possible combinations for this map component up to total order 10
                    comblist = list(itertools.combinations_with_replacement(
                        np.arange(k+1),#[-1:],
                        o+1))
                    
                    # Go through each of these combinations
                    for entry in comblist:
                        
                        # Create a transport map component term
                        if k in entry:
                            
                            if entry == (k,):
                                
                                monotone[-1].append(
                                    [])
                                
                            else:
                            
                                # Level 2: Specify the polynomial order of this term;
                                # It's a Hermite function
                                monotone[-1].append(
                                    list(entry)+['HF'])
                                
        # Delete any map object which might already exist
        if "tm" in globals():
            del tm
            
        # Parameterize the transport map
        tm     = transport_map(
            monotone                = monotone,
            nonmonotone             = nonmonotone,
            X                       = copy.copy(X),         # Training ensemble
            polynomial_type         = "hermite function",   # We use Hermite functions for stability
            monotonicity            = "integrated rectifier", # Because we have cross-terms, we require the integrated rectifier formulation
            standardize_samples     = True,                 # Standardize X before training
            workers                 = 1,                    # Number of workers for the parallel optimization; 1 is not parallel
            quadrature_input        = {                     # Keywords for the Gaussian quadrature used for integration
                'order'         : 25,       # If the map is bad, increase this number; takes more computational effort
                'adaptive'      : False,
                'threshold'     : 1E-9,
                'verbose'       : False,
                'increment'     : 6})
        
        # Train the map
        if 'dict_coeffs_order='+str(maxorder)+'_complexity='+str(complexity)+'.p' not in os.listdir(root_directory):
            
            print('No pre-optimized coefficients found. Optimizing...')
            
            tm.optimize()
        
            dict_coeffs = {
                'coeffs_mon'    : tm.coeffs_mon,
                'coeffs_nonmon' : tm.coeffs_nonmon}
            pickle.dump(dict_coeffs,open('dict_coeffs_order='+str(maxorder)+'_complexity='+str(complexity)+'.p','wb'))
            
        else:
            
            print('Pre-optimized coefficients found. Extracting...')
            
            dict_coeffs = pickle.load(open('dict_coeffs_order='+str(maxorder)+'_complexity='+str(complexity)+'.p','rb'))
            
            tm.coeffs_mon       = copy.copy(dict_coeffs['coeffs_mon'])
            tm.coeffs_nonmon    = copy.copy(dict_coeffs['coeffs_nonmon'])
            
        dct     = {
            'coeffs_mon'    : tm.coeffs_mon,
            'coeffs_nonmon' : tm.coeffs_nonmon,
            'X'             : tm.X,
            'X_mean'        : tm.X_mean,
            'X_std'         : tm.X_std,
            'monotone'      : tm.monotone,
            'nonmonotone'   : tm.nonmonotone}
        
        pickle.dump(dct,open('TM_complexity='+str(complexity)+'_maxorder='+str(maxorder)+'.p','wb'))
