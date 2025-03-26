import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import copy
import matplotlib
import scipy.spatial

plt.close('all')


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


def f1(x):
    
    # return np.exp( -1*x - 2*x**2 + 2*x**3)
    return scipy.stats.norm.pdf(x=x,loc=0.25,scale=0.1) + scipy.stats.norm.pdf(x=x,loc=0.75,scale=0.1)

def f2(x):
    

    # return np.exp( 1*x - 2*x**2 - 0.4*x**3) + scipy.stats.norm.pdf(x=y,loc=0.25,scale=0.075) + scipy.stats.norm.pdf(x=y,loc=0.75,scale=0.075) # 2.5*y - 2*y**2 + 0.5*y**3

    return np.exp( 1*x - 0.2*x**2 - 0.4*x**3)


def f3(x):
    
    return np.exp( 567*x**4 - 1102*x**3 + 677.5*x**2 - 136*x + 1.9375)

resolution  = 500

X,Y     = np.meshgrid(
    np.linspace(0,1,resolution),
    np.linspace(0,1,resolution))

XY      = np.column_stack((
    np.ndarray.flatten(X),
    np.ndarray.flatten(Y) ))


Z1  = GaussQuadrature(
    f               = f1,
    a               = 0,
    b               = XY[:,0]) + XY[:,0]*0.75

Z1  -= np.min(Z1)
Z1  /= np.max(Z1)

Z1b = GaussQuadrature(
    f               = f2,
    a               = 0,
    b               = XY[:,0])*15 + 38.6*XY[:,1]**3 - 56.35*XY[:,1]**2 + 13.2125*XY[:,1] - 0.853125

Z2  = GaussQuadrature(
    f               = f2,
    a               = 0,
    b               = XY[:,1])*15 + 38.6*XY[:,0]**3 - 56.35*XY[:,0]**2 + 13.2125*XY[:,0] - 0.853125

Z2  -= np.min(Z2)
Z2  /= np.max(Z2)

Z3  = GaussQuadrature(
    f               = f2,
    a               = 0,
    b               = XY[:,0])*15 + 56.7*XY[:,1]**4 - 90.2*XY[:,1]**3 + 67.75*XY[:,1]**2 - 13.6*XY[:,1]

Z3  -= np.min(Z3)
Z3  /= np.max(Z3)

Z2norm  = copy.copy(Z2)
Z2norm  -= np.min(Z2norm)
Z2norm  /= np.max(Z2norm)

cmap = matplotlib.cm.get_cmap('turbo')

cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:goldenrod",
     "xkcd:orangish red"])

cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:sky blue",
     "xkcd:cerulean"])

plt.figure(figsize=(10,10))

gs  = matplotlib.gridspec.GridSpec(nrows = 2,ncols = 2)

plt.subplot(gs[0,0])


x   = np.linspace(0,1,101)
y   = GaussQuadrature(
    f               = f1,
    a               = 0,
    b               = x ) + x*0.75
y   -= np.min(y)
y   /= np.max(y)
for i in range(len(x)-1):
    plt.plot(x[i:i+2],y[i:i+2],color=cmap(np.mean(y[i:i+2])))
plt.gca().set_xticks([0,1])
plt.gca().set_xticklabels(['-','+'])
plt.gca().set_yticks([0,1])
plt.gca().set_yticklabels(['-','+'])
plt.xlabel('input $x$', labelpad=-15)
plt.ylabel('output $z=S(x)$', labelpad=-15)

# Plot the inverse model arrow
plt.plot([0,x[50]+0.0075],[y[50]+0.01,y[50]+0.01],zorder=-2,color='xkcd:grey',label='inverse $S^{-1}(x)$')
plt.arrow(x[50]+0.0075,y[50]+0.0075,0,-y[50]+0.0375,zorder=-1,head_width = 0.02,ec='xkcd:grey',fc='xkcd:grey',width=0.005)

plt.title("$\mathbf{A}:$ monotone function", loc = "left")

plt.subplot(gs[0,1])
x   = np.linspace(0,1,101)
y   = GaussQuadrature(
    f               = f1,
    a               = 0,
    b               = x ) - x*1.00
y   -= np.min(y)
y   /= np.max(y)
for i in range(len(x)-1):
    plt.plot(x[i:i+2],y[i:i+2],color=cmap(np.mean(y[i:i+2])))
plt.gca().set_xticks([0,1])
plt.gca().set_xticklabels(['-','+'])
plt.gca().set_yticks([0,1])
plt.gca().set_yticklabels(['-','+'])
plt.xlabel('input $x$', labelpad=-15)
plt.ylabel('output $z=S(x)$', labelpad=-15)

plt.plot([0,x[36]+0.0075],[y[49]+0.01,y[49]+0.01],zorder=-2,color='xkcd:grey')
plt.plot([x[36]+0.0075,x[64]+0.0075],[y[49]+0.01,y[49]+0.01],zorder=-2,color='xkcd:grey',linestyle='--')

plt.plot([x[48]+0.0075]*2,[y[49]+0.0075,0.1],zorder=-1,color='xkcd:grey',linestyle='--')
plt.plot([x[36]+0.0075]*2,[y[36]+0.0075,0.1],zorder=-1,color='xkcd:grey',linestyle='--')
plt.plot([x[64]+0.0075]*2,[y[64]+0.0075,0.1],zorder=-1,color='xkcd:grey',linestyle='--')

plt.arrow(x[48]+0.0075,0.1,0,-0.05,zorder=-1,ec='xkcd:grey',fc='xkcd:grey',head_width = 0.02,width=0.005)
plt.arrow(x[36]+0.0075,0.1,0,-0.05,zorder=-1,ec='xkcd:grey',fc='xkcd:grey',head_width = 0.02,width=0.005)
plt.arrow(x[64]+0.0075,0.1,0,-0.05,zorder=-1,ec='xkcd:grey',fc='xkcd:grey',head_width = 0.02,width=0.005)

plt.gca().text(0.42, 0.1, '$\mathbf{?}$', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:grey')

plt.gca().text(0.525, 0.1, '$\mathbf{?}$', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:grey')

plt.gca().text(0.675, 0.1, '$\mathbf{?}$', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:grey')

plt.title("$\mathbf{B}:$ nonmonotone function", loc = "left")

plt.subplot(gs[1,0])
ax  = plt.gca()
cm2 = plt.contourf(X,Y,Z1.reshape((resolution,resolution)),31,cmap=cmap,vmin = np.min(Z1) - (np.max(Z1)-np.min(Z1))*0.2)
plt.axis('equal')
plt.xlabel('dimension $x_1$', labelpad=-15)
plt.ylabel('dimension $x_2$', labelpad=-15)
ax.set_xticks([0,1])
ax.set_xticklabels(['-','+'])
ax.set_yticks([0,1])
ax.set_yticklabels(['-','+'])

plt.title("$\mathbf{C}:$ monotone $S_{1}(x_{1})$", loc = "left")

plt.gca().annotate('', xy=(0.3, 0.7), xycoords='axes fraction', xytext=(0.3, 0.3), 
                    arrowprops=dict(color="xkcd:dark grey",headlength=10,headwidth=10,width=3),)
plt.gca().annotate('', xy=(0.7, 0.3), xycoords='axes fraction', xytext=(0.3, 0.3), 
                    arrowprops=dict(color="xkcd:dark grey",headlength=10,headwidth=10,width=3),)

plt.text(
    0.5,
    0.275,
    "monotone in $x_1$",
    ha  = "center",
    va  = "top",
    transform = plt.gca().transAxes)

plt.text(
    0.275,
    0.5,
    "constant in $x_2$",
    ha  = "right",
    va  = "center",
    rotation = 90,
    transform = plt.gca().transAxes)

plt.subplot(gs[1,1])
ax  = plt.gca()
cm1 = plt.contourf(X,Y,Z2.reshape((resolution,resolution)),31,cmap=cmap,vmin = np.min(Z2) - (np.max(Z2)-np.min(Z2))*0.2)
plt.axis('equal')
plt.xlabel('dimension $x_1$', labelpad=-15)
plt.ylabel('dimension $x_2$', labelpad=-15)
ax.set_xticks([0,1])
ax.set_xticklabels(['-','+'])
ax.set_yticks([0,1])
ax.set_yticklabels(['-','+'])

plt.title("$\mathbf{D}:$ monotone $S_{2}(x_{1},x_{2})$", loc = "left")

plt.gca().annotate('', xy=(0.3, 0.7), xycoords='axes fraction', xytext=(0.3, 0.3), 
                    arrowprops=dict(color="xkcd:dark grey",headlength=10,headwidth=10,width=3),)
plt.gca().annotate('', xy=(0.7, 0.3), xycoords='axes fraction', xytext=(0.3, 0.3), 
                    arrowprops=dict(color="xkcd:dark grey",headlength=10,headwidth=10,width=3),)

plt.text(
    0.5,
    0.275,
    "nonmonotone in $x_1$",
    ha  = "center",
    va  = "top",
    transform = plt.gca().transAxes)

plt.text(
    0.275,
    0.5,
    "monotone in $x_2$",
    ha  = "right",
    va  = "center",
    rotation = 90,
    transform = plt.gca().transAxes)

# Save the figure
plt.savefig('monotonicity.png',dpi=600,bbox_inches='tight')
plt.savefig('monotonicity.pdf',dpi=600,bbox_inches='tight')