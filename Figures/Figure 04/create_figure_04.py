import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d, UnivariateSpline

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.close('all')

"""
Assume we have a monotonic scalar function g: R -> R such that

f_Y(y) = f_X(g^{-1}(y)) | d/(dy) g^{-1}(y) |

Which, using y = g(x) or x = g^{-1}(y), can also be written as:
    
f_Y(y) = f_X(x) |  1/(d/(dy)*g(y)) |

"""

y       = np.linspace(-2,2,1001)**3 + np.linspace(-2,2,1001)/2
x       = np.linspace(-2,2,1001)

f       = interp1d(x,y,fill_value="extrapolate")
finv    = interp1d(y,x,fill_value="extrapolate")



def dfdx(x):
    return (f(x+1E-9)-f(x))/1E-9

def dfinvdx(x):
    return (finv(x+1E-9)-finv(x))/1E-9

target  = scipy.stats.norm.pdf(f(x))*np.abs(dfdx(x))

tarpdf  = interp1d(x,target,fill_value="extrapolate")

#%%

# Calculate the bin for the target
x_bin       = np.linspace(-0.7,0.,1001)
tar_bin     = tarpdf(x_bin)

# Calculate the bin for the reference
z_bin       = f(x_bin)
term1_bin   = tarpdf(finv(z_bin))
term2_bin   = np.abs(dfinvdx(z_bin))
ref_bin     = term1_bin*term2_bin

#%%

# Increments
increments      = 101
increment_width = 0.5

increment_start = np.linspace(-2,2-increment_width,increments)

i       = 36

plt.close("all")


plt.figure(figsize=(12,6))



gs_super = GridSpec(
    nrows           = 1,
    ncols           = 3,
    hspace          = 0,
    wspace          = 0.6,
    width_ratios    = [1.,0.2,0.2])

from matplotlib import gridspec

gs = gridspec.GridSpecFromSubplotSpec(
    nrows           = 2,
    ncols           = 2,
    hspace          = 0.1,
    wspace          = 0.1,
    height_ratios   = [0.2,1.0],
    width_ratios    = [1.0,0.2],
    subplot_spec    = gs_super[0,0])




#%%

plt.subplot(gs[0,0])


# Calculate the target pdf

x       = np.linspace(-2,2,101)


target  = [0]+list(tarpdf(x))+[0]
x       = [-2]+list(x)+[2]


plt.fill(
    x,
    target,
    color   = "xkcd:orangish red",
    alpha   = 0.5,
    lw      = 0)


plt.fill(
    [x_bin[0]]+list(x_bin)+[x_bin[-1]],
    [0]+list(tar_bin)+[0],
    color   = "xkcd:orangish red",
    lw      = 0)


plt.ylim(0,np.max(target)*1.1)
plt.xlim(-2,2)


plt.gca().invert_yaxis()

# Remove all axis ticks
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)


plt.gca().text(0.415, 1.025, '$\pi$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:orangish red')

plt.gca().text(0.45, 1.025, '$ ($ \,$)$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='k')

plt.gca().text(0.4625, 1.025, '$\mathbf{x}$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:orangish red')


plt.gca().invert_yaxis()


#%%


plt.subplot(gs[1,0])

plt.plot(
    x,
    f(x),
    color   = 'xkcd:dark grey')

plt.xlim(-2,2)
plt.ylim(-2,2)


plt.fill(
    [x_bin[0]]+list(x_bin)+[x_bin[-1]],
    [2]+list(f(x_bin))+[2],
    color   = "xkcd:orangish red",
    alpha   = 0.5,
    lw      = 0)

plt.fill(
    [2]+list(x_bin)+[2],
    [f(x_bin[0])]+list(f(x_bin))+[f(x_bin[-1])],
    color   = "xkcd:grass green",
    alpha   = 0.5,
    lw      = 0)



# Remove all axis ticks
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.gca().text(0.725, 0.6, '$\mathbf{S}(\mathbf{x})$', transform=plt.gca().transAxes, fontsize=14,
        verticalalignment='center',horizontalalignment='center',color='xkcd:dark grey')





#%%

 
plt.subplot(gs[1,1])


z       = np.linspace(-2,2,1001)

term1   = tarpdf(finv(z))
term2   = np.abs(dfinvdx(z))

ref     = scipy.stats.norm.pdf(z)
# ref     = term1*term2




plt.fill(
    [0]+list(ref)+[0],
    [-2]+list(z)+[2],
    color   = "xkcd:grass green",
    alpha   = 0.5,
    lw      = 0)


plt.fill(
    [0]+list(ref_bin)+[0],
    [z_bin[0]]+list(z_bin)+[z_bin[-1]],
    color   = "xkcd:grass green",
    lw      = 0)


plt.xlim(0,np.max(ref)*1.1)
plt.ylim(-2,2)


# Remove all axis ticks
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)



plt.gca().text(0.275, 1.025, '$\eta$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:grass green')

plt.gca().text(0.45, 1.025, '$($ \,$)$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='k')

plt.gca().text(0.535, 1.025, '$\mathbf{z}$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:grass green')


#%%

# Second subplot: density map

gs2 = gridspec.GridSpecFromSubplotSpec(
    nrows           = 2,
    ncols           = 1,
    hspace          = 0.1,
    wspace          = 0.1,
    height_ratios   = [0.2,1.0],
    subplot_spec    = gs_super[0,1])

plt.subplot(gs2[1,0])


plt.fill(
    [0]+list(term1)+[0],
    [-2]+list(z)+[2],
    lw      = 0,
    alpha   = 0.5,
    color   = "xkcd:grass green")


plt.fill(
    [0]+list(term1_bin)+[0],
    [z_bin[0]]+list(z_bin)+[z_bin[-1]],
    color   = "xkcd:grass green",
    lw      = 0)

# Remove all axis ticks
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.gca().text(-0., 1.025, '$\pi$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:orangish red')

plt.gca().text(0.2, 1.025, '$(S^{-1}($ \,$))$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='k')

plt.gca().text(0.725, 1.025, '$\mathbf{z}$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:grass green')

plt.gca().text(-0.75, 0.5, '$=$', transform=plt.gca().transAxes, fontsize=36,
        verticalalignment='center',horizontalalignment='center',color='xkcd:grass green')

plt.xlim(0,np.max(term1)*1.1)
plt.ylim(-2,2)


#%%

# Third subplot: density map

gs3 = gridspec.GridSpecFromSubplotSpec(
    nrows           = 2,
    ncols           = 1,
    hspace          = 0.1,
    wspace          = 0.1,
    height_ratios   = [0.2,1.0],
    subplot_spec    = gs_super[0,2])

plt.subplot(gs3[1,0])

plt.fill(
    [0]+list(term2)+[0],
    [-2]+list(z)+[2],
    color   = "xkcd:grass green",
    alpha   = 0.5,
    lw      = 0)

plt.fill(
    [0]+list(term2_bin)+[0],
    [z_bin[0]]+list(z_bin)+[z_bin[-1]],
    color   = "xkcd:grass green",
    lw      = 0)

plt.xlim(0,np.max(term2)*1.1)
plt.ylim(-2,2)

# Remove all axis ticks
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.gca().text(0.175, 1.012, '$\mathbf{z}$', transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:grass green')

plt.gca().text(0.025, 1.02, '$|\partial$  $S^{-1}($ \,$)|$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='k')

plt.gca().text(0.725, 1.025, '$\mathbf{z}$', transform=plt.gca().transAxes, fontsize=16,
        verticalalignment='bottom',horizontalalignment='left',color='xkcd:grass green')



plt.gca().text(-0.75, 0.5, '×', transform=plt.gca().transAxes, fontsize=36,
        verticalalignment='center',horizontalalignment='center',color='xkcd:grass green')

plt.savefig('change_of_variables.png',dpi=600,bbox_inches='tight')
plt.savefig('change_of_variables.pdf',dpi=600,bbox_inches='tight')