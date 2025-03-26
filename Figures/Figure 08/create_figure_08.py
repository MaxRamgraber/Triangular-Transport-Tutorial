import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.gridspec import GridSpec
import matplotlib

# cmap = matplotlib.cm.get_cmap('turbo')
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:cerulean",
     "xkcd:grass green",
     "xkcd:goldenrod",
     "xkcd:orangish red"])

plt.close("all")

#%%

def GaussQuadrature(f, a, b, order = 100, args = None, Ws = None, xis = None, 
                    adaptive = False, threshold = 1E-6, increment = 1, 
                    verbose = False, full_output = False):
    
    """
    This function implements Gaussian quadrature, using the following input:
        
        f           : element-wise function to be integrated
        a           : lower integration bound, scalar or vector
        b           : upper integration bound, scalar or vector
        order       : order of the Legendre polynomial, integer
        args        : supporting arguments passed to the function, tuple
        
    To speed up computation, the integration points and weights can be computed
    in advance. In this case, both can be provided as:
        
        Ws          : weights of the integration points, vector
        xis         : positions of the integration points, vector
        
    If adaptive quadrature is desired, there is a flag which increases the
    order of the polynomial until the difference falls below the threshold:
        
        adaptive    : flag for the adaptive procedure, boolean
        threshold   : threshold for the difference in the adaptive integration,
                      adaptation stops after difference in integration result
                      falls below this value, scalar
        increment   : increment by which the order is increased in each 
                      adaptation cycle, scalar
        verbose     : flag for printing information
        
    Finally, if we wish to return the positions and weights of the integration
    points, there is a flag which not only returns the integration results, but
    also the identified order, position, and weights of the integration points:
        
        full_output : flag for detailed output, returns a tuple with (results,
                      order,xis,Ws); if False, returns only results; boolean
    
    """
    
    import numpy as np
    import copy
    
    # =========================================================================
    # Here the actual magic starts
    # =========================================================================
    
    # If adaptation is desired, we must iterate; prepare a flag for this 
    repeat      = True
    iteration   = 0
    
    # Iterate, if adaptation = True; Otherwise, iteration stops after one round
    while repeat:
        
        # Increment the iteration counter
        iteration   += 1
    
        # If required, determine the weights and positions of the integration
        # points; always required if adaptation is active
        if Ws is None or xis is None or adaptive == True:
            
            # Weights and integration points are not specified; calculate them
            # To get the weights and positions of the integration points, we must
            # provide the *order*-th Legendre polynomial and its derivative
            # As a first step, get the coefficients of both functions
            coefs       = np.zeros(order+1)
            coefs[-1]   = 1
            coefs_der   = np.polynomial.legendre.legder(coefs)
            
            # With the coefficients defined, define the Legendre function
            LegendreDer = np.polynomial.legendre.Legendre(coefs_der)
            
            # Obtain the locations of the integration points
            xis = np.polynomial.legendre.legroots(coefs)
            
            # Calculate the weights of the integration points
            Ws  = 2.0/( (1.0-xis**2)*(LegendreDer(xis)**2) )
            
        # If any of the boundaries is a vector, vectorize the operation
        if not np.isscalar(a) or not np.isscalar(b):
            
            # If only one of the bounds is a scalar, vectorize it
            if np.isscalar(a) and not np.isscalar(b):
                a       = np.ones(b.shape)*a
            if np.isscalar(b) and not np.isscalar(a):
                b       = np.ones(a.shape)*b
            

            # Alternative approach, more amenable to dimension-sensitivity in
            # the function f. To speed up computation, pre-calculate the limit
            # differences and sum
            lim_dif = b-a
            lim_sum = b+a
            result  = np.zeros(a.shape)
            
            # print('limdifshape:'+str(lim_dif.shape))
            # print('resultshape:'+str(result.shape))
            
            # =============================================================
            # To understand what's happening here, consider the following:
            # 
            # lim_dif and lim_sum   - shape (N)
            # funcres               - shape (N) up to shape (N-by-C-by-C)
            
            # If no additional arguments were given, simply call the function
            if args is None:
                
                result  = lim_dif*0.5*(Ws[0]*f(lim_dif*0.5*xis[0] + lim_sum*0.5))
                
                for i in np.arange(1,len(Ws)):
                    result  += lim_dif*0.5*(Ws[i]*f(lim_dif*0.5*xis[i] + lim_sum*0.5))
                    
            # Otherwise, pass the arguments on as well
            else:
                
                funcres     = f(
                    lim_dif*0.5*xis[0] + lim_sum*0.5,
                    *args)
                
                # =========================================================
                # Depending on what shape the output function returns, we 
                # must take special precautions to ensure the product works
                # =========================================================
                
                # If the function output is the same size as its input
                if len(funcres.shape) == len(lim_dif.shape):
                    
                    result  = lim_dif*0.5*(Ws[0]*funcres)
                    
                    for i in np.arange(1,len(Ws)):
                        
                        funcres     = f(
                            lim_dif*0.5*xis[i] + lim_sum*0.5,
                            *args)
                        
                        result  += lim_dif*0.5*(Ws[i]*funcres)
                        
                        
                        
                        
                        
                
                # If the function output has one dimension more than its
                # corresponding input
                elif len(funcres.shape) == len(lim_dif.shape)+1:
                    
                    result  = np.einsum(
                        'i,ij->ij',
                        lim_dif*0.5*Ws[0],
                        funcres)
                    
                    for i in np.arange(1,len(Ws)):
                        
                        funcres     = f(
                            lim_dif*0.5*xis[i] + lim_sum*0.5,
                            *args)
                        
                        result  += np.einsum(
                            'i,ij->ij',
                            lim_dif*0.5*Ws[i],
                            funcres)
                
                    # # result  = lim_dif*0.5*(Ws[0]*funcres)
                    # result  = np.moveaxis(
                    #     lim_dif*0.5*Ws[0]*np.moveaxis(
                    #         funcres,-1,0),0,-1)
                    
                    # for i in np.arange(1,len(Ws)):
                        
                    #     funcres     = f(
                    #         lim_dif*0.5*xis[i] + lim_sum*0.5,
                    #         *args)
                        
                    #     result  += np.moveaxis(
                    #     lim_dif*0.5*Ws[i]*np.moveaxis(
                    #         funcres,-1,0),0,-1)
                        
                # If the function output has one dimension more than its
                # corresponding input
                elif len(funcres.shape) == len(lim_dif.shape)+2:
                    
                    result  = np.einsum(
                        'i,ijk->ijk',
                        lim_dif*0.5*Ws[0],
                        funcres)
                    
                    for i in np.arange(1,len(Ws)):
                        
                        funcres     = f(
                            lim_dif*0.5*xis[i] + lim_sum*0.5,
                            *args)
                        
                        result  += np.einsum(
                            'i,ijk->ijk',
                            lim_dif*0.5*Ws[i],
                            funcres)
                    
                else:
                    
                    raise Exception('Shape of input dimension is '+\
                    str(lim_sum.shape)+' and shape of output dimension is '+\
                    str(funcres.shape)+'. Currently, we have only implemented '+\
                    'situations in which input and output are the same shape, '+\
                    'or where output is one or two dimensions larger.')

        else:
                
            # Now start the actual computation.
            
            # If no additional arguments were given, simply call the function
            if args is None:
                result  = (b-a)*0.5*np.sum( Ws*f( (b-a)*0.5*xis+ (b+a)*0.5 ) )
            # Otherwise, pass the arguments on as well
            else:
                result  = (b-a)*0.5*np.sum( Ws*f(
                    (b-a)*0.5*xis + (b+a)*0.5,
                    *args) )
            
        # if adaptive, store results for next iteration
        if adaptive:
            
            # In the first iteration, just store the results
            if iteration == 1:
                previous_result = copy.copy(result)
            
            # In later iterations, check integration process
            else:
                
                # How much did the results change?
                change          = np.abs(result-previous_result)
            
                # Check if the change in results was sufficient
                if np.max(change) < threshold or iteration > 1000:
                    
                    # Stop iterating
                    repeat      = False
                    
                    if iteration > 1000:
                        print('WARNING: Adaptive integration stopped after '+\
                        '1000 iteration cycles. Final change: '+str(change))
                            
                    # Print the final change if required
                    if verbose:
                        print('Final maximum change of Gauss Quadrature: ' + \
                              str(np.max(change)))
                            
            # If we must still continue repeating, increment order and store
            # current result for next iteration
            if repeat:
                order           += increment
                previous_result = copy.copy(result)
          
        # If no adaptation is required, simply stop iterating
        else:
            repeat  = False
        
    # If full output is desired
    if full_output:
        result  = (result,order,xis,Ws)
        
    if verbose:
        print('Order: '+str(order))
    
    return result

#%%

# Create an empty figure
plt.figure(figsize=(12,3))

# Add GridSpec
gs = GridSpec(
    nrows           = 1,
    ncols           = 3,
    wspace          = 0.3)

# Create some nonmonotone function
def f1(x):
    #return scipy.stats.norm.pdf(x=x,loc=0.25,scale=0.1) + scipy.stats.norm.pdf(x=x,loc=0.75,scale=0.1)
    return 0.2*x**4 - 4*x**3 + 4*x

def expf1(x):
    #return scipy.stats.norm.pdf(x=x,loc=0.25,scale=0.1) + scipy.stats.norm.pdf(x=x,loc=0.75,scale=0.1)
    return np.exp(f1(x))


x   = np.linspace(-1,1,1001)

y   = f1(x)
colorlist = []
for i in range(len(x)-1):
    colorlist.append(cmap((np.mean(y[i:i+2])-np.min(y))/(np.max(y) - np.min(y))))

# =============================================================================
# Plot the integrated rectified function
# =============================================================================


plt.subplot(gs[0,2])
plt.title(r'$\bf{C}$: + integration', loc='left')

y   = GaussQuadrature(
    f               = expf1,
    a               = -1,
    b               = x)

for i in range(len(x)-1):
    plt.plot(x[i:i+2],y[i:i+2],color=colorlist[i])

plt.ylabel('monotone $\int_{0}^{x} \exp(\hat{g}(\gamma)) d \gamma$')
plt.gca().set_yticks([0.,1.,2.,3.])
plt.gca().set_yticklabels(["0","1","2","3"])

plt.xlabel("$x$")
plt.gca().set_xticks([])
plt.gca().set_xticklabels([])
# plt.gca().tick_params(axis="y",direction="in", pad=-22)

# -----------------------------------------------------------------------------
# Arrow right
# -----------------------------------------------------------------------------

# Fudge together a filled color gradient
for k in range(101):
    
    xp  = -0.3 + 0.3*k/100 # 0.115*k/100
    
    col     = np.asarray(matplotlib.colors.to_rgba('xkcd:grey')) + (np.asarray(matplotlib.colors.to_rgba('xkcd:silver')) - np.asarray(matplotlib.colors.to_rgba('xkcd:grey')))*k/100
    
    if k < 100:
        plt.gca().annotate('', xy=(xp, 0.45), xycoords='axes fraction', xytext=(xp,0.55), 
                            arrowprops=dict(color=col,headlength=1,headwidth=0,width=1),zorder=-1)
        
        
        # plt.gca().annotate('', xy=(0.42, yp), xycoords='axes fraction', xytext=(0.48, yp), 
        #                     arrowprops=dict(color=col,headlength=1,headwidth=0,width=1))
        
plt.gca().annotate('', xy=(xp+0.08, 0.5), xycoords='axes fraction', xytext=(xp+0.08-0.001, 0.5), 
                    arrowprops=dict(color=col,headlength=20,headwidth=35,width=1),)













    
# =============================================================================
# Plot the rectified function
# =============================================================================


plt.subplot(gs[0,1])
plt.title(r'$\bf{B}$: + rectifier', loc='left')

y   = expf1(x)

for i in range(len(x)-1):
    plt.plot(x[i:i+2],y[i:i+2],color=colorlist[i])
    
plt.ylabel('positive $\exp(\hat{g}(x))$')
plt.gca().set_yticks([0.,1.,2.,3.,4.])
plt.gca().set_yticklabels(["0","1","2","3","4"])

plt.xlabel("$x$")
plt.gca().set_xticks([])
plt.gca().set_xticklabels([])

# -----------------------------------------------------------------------------
# Arrow right
# -----------------------------------------------------------------------------

# Fudge together a filled color gradient
for k in range(101):
    
    xp  = -0.3 + 0.3*k/100 # 0.115*k/100
    
    col     = np.asarray(matplotlib.colors.to_rgba('xkcd:grey')) + (np.asarray(matplotlib.colors.to_rgba('xkcd:silver')) - np.asarray(matplotlib.colors.to_rgba('xkcd:grey')))*k/100
    
    if k < 100:
        plt.gca().annotate('', xy=(xp, 0.45), xycoords='axes fraction', xytext=(xp,0.55), 
                            arrowprops=dict(color=col,headlength=1,headwidth=0,width=1),zorder=-1)
        
        
        # plt.gca().annotate('', xy=(0.42, yp), xycoords='axes fraction', xytext=(0.48, yp), 
        #                     arrowprops=dict(color=col,headlength=1,headwidth=0,width=1))
        
plt.gca().annotate('', xy=(xp+0.08, 0.5), xycoords='axes fraction', xytext=(xp+0.08-0.001, 0.5), 
                    arrowprops=dict(color=col,headlength=20,headwidth=35,width=1),)


# =============================================================================
# Plot the pre-monotone function f^{pre}
# =============================================================================

y   = f1(x)

plt.subplot(gs[0,0])
plt.title(r'$\bf{A}$: nonmonotone function', loc='left')

for i in range(len(x)-1):
    plt.plot(x[i:i+2],y[i:i+2],color=colorlist[i])
    
plt.ylabel('nonmonotone $\hat{g}(x)$')
plt.gca().set_yticks([-1.,0.,1.])
plt.gca().set_yticklabels(["-1","0","1"])
# plt.gca().tick_params(axis="y",direction="in", pad=-22)

plt.xlabel("$x$")
plt.gca().set_xticks([])
plt.gca().set_xticklabels([])
    



plt.savefig('integrated_rectifier_v3.png',dpi=600,bbox_inches='tight')
plt.savefig('integrated_rectifier_v3.pdf',dpi=600,bbox_inches='tight')
