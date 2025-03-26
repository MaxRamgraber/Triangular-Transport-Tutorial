# Load in a number of libraries we will use
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats
import copy
import scipy.optimize
import time
import pickle
import os
import itertools
from matplotlib.colors import LinearSegmentedColormap

vcolor  = "#5FBFEF"#'xkcd:sky blue'
hcolor  = "#387FA6"#'xkcd:steel blue'

cmap = LinearSegmentedColormap.from_list("maxturbo", ["#285A76",hcolor,vcolor,"#94D4F4"])

xlims   = [-0.75,0.75]
ylims   = [0,1]

# def fun(x):
    
#     xl  = x*2 - 1
    
#     return 2*xl**3 - 1*xl

def prior_pdf(x):
    
    pdf     = np.zeros(x.shape)
    
    # pdf     = scipy.stats.norm.pdf(loc=0.2,scale=0.05,x=x)
    # pdf     += scipy.stats.norm.pdf(loc=0.5,scale=0.05,x=x)
    # pdf     += scipy.stats.norm.pdf(loc=0.8,scale=0.25,x=x)
    
    
    pdf     = scipy.stats.beta.pdf(a = 15, b = 7, x = x)
    pdf     += scipy.stats.beta.pdf(a = 3, b = 8, x = x)
    
    return pdf

plt.close("all")

# coeffs_fun = np.asarray([-1,10,-24,16])

# coeffs_fun = np.asarray([-1,12,-20,15])

coeffs_fun = np.asarray([-3,1,0,20])
funpoly = np.polynomial.Polynomial(coeffs_fun)


plt.figure(figsize=(16,4.8))

gs  = GridSpec(
    nrows           = 1,
    ncols           = 3,
    wspace          = 0.2,
    width_ratios    = [0.95,1,1])

x   = np.linspace(0,1,101)
y   = prior_pdf(x)

prior = lambda x: scipy.stats.norm.pdf(
    x       = x,
    loc     = 0.5,
    scale   = 0.2)

logprior = lambda x: scipy.stats.norm.logpdf(
    x       = x,
    loc     = 0.5,
    scale   = 0.2)

#%%

gs2 = gs[0].subgridspec(
    nrows   = 2,
    ncols   = 2,
    hspace  = 0.,
    wspace  = 0.,
    height_ratios   = [1,0.2],
    width_ratios    = [0.2,1.0])

plt.subplot(gs2[0,1])

plt.plot(x,funpoly(x),ls='--',color="#666666")

xlims = [0,1]
ylims = plt.gca().get_ylim()
ylims = [ylims[0]-2.5,ylims[1]]

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)

plt.xlabel("input variable $x$")
plt.ylabel("output variable $y=f(x)$")

plt.title("$\mathbf{A}:$ Deterministic function",loc="left")

# Plot the function line
plt.plot(
    [0.75,0.75,0.],
    [ylims[0]]+list(funpoly(np.asarray([0.75,0.75]))),
    color = "xkcd:cobalt",
    zorder = -1,
    ls = ":")

plt.text(
    0.775,
    0,
    "$x^{*}$",
    ha = "left",
    va = "center",
    color = "xkcd:cobalt",
    fontsize=10)

plt.text(
    0.4,
    funpoly(0.75)+0.025,
    "$f(x^{*})$",
    ha = "center",
    va = "bottom",
    color = "xkcd:cobalt",
    fontsize=10)


#%%

gs2 = gs[1].subgridspec(
    nrows   = 2,
    ncols   = 2,
    hspace  = 0.,
    wspace  = 0.,
    height_ratios   = [1,0.2],
    width_ratios    = [0.2,1.0])





der_coeffs_fun = np.polynomial.polynomial.polyder(coeffs_fun)
der_funpoly = np.polynomial.Polynomial(der_coeffs_fun)

ypts    = np.linspace(ylims[0],ylims[1],1001)

xpts    = []
for ypt in ypts:
    
    # Invert the function, first find the polynomial coefficient for root finding
    loc_coeffs = coeffs_fun - np.asarray([ypt,0,0,0])
    
    # Find the roots
    roots = np.polynomial.Polynomial(loc_coeffs).roots()
    
    # Only keep the real roots
    roots = roots[np.where(roots.imag == 0.)]
    roots = [r.real for r in roots]
    
    xpts .append( np.sum([prior(r) / np.abs(der_funpoly(r)) for r in roots] ) )






plt.subplot(gs2[0,1])

# plt.plot(x,funpoly(x),ls='--',color="#666666")

for idx in range(len(x)-1):
    
    plt.plot(
        x[idx:idx+2],
        funpoly(x[idx:idx+2]),
        color = cmap(np.mean(prior(x[idx:idx+2])/prior(0.5))))



plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)


# Plot the function line
plt.plot(
    [0.75,0.75,0.],
    [ylims[0]]+list(funpoly(np.asarray([0.75,0.75]))),
    color = "xkcd:cobalt",
    zorder = -1,
    ls = ":")

plt.text(
    0.775,
    0,
    "$x^{*}$",
    ha = "left",
    va = "center",
    color = "xkcd:cobalt",
    fontsize=10)

plt.text(
    0.4,
    funpoly(0.75)+0.025,
    "$f(x^{*})$",
    ha = "center",
    va = "bottom",
    color = "xkcd:cobalt",
    fontsize=10)



plt.title("$\mathbf{B}:$ Deterministic coupling",loc="left")

plt.subplot(gs2[1,1])



y_prior = prior(x)

plt.fill(
    [0]+list(x)+[1],
    [0]+list(y_prior)+[0],
    facecolor = hcolor)


plt.gca().set_xlim(xlims)

plt.gca().invert_yaxis()
plt.gca().set_ylim([plt.gca().get_ylim()[0]*1.1,0])

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.xlabel("marginal random variable $x$")
plt.ylabel("$p(x)$")

plt.subplot(gs2[0,0])


y_prior = prior(x)

plt.fill(
    [0]+list(xpts)+[0],
    [ypts[0]]+list(ypts)+[ypts[-1]],
    facecolor = vcolor)


# plt.gca().set_xlim(xlims)

plt.gca().invert_xaxis()


plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.gca().set_ylim(ylims)
plt.gca().set_xlim([plt.gca().get_xlim()[0]*1.1,0])

plt.ylabel("marginal random variable $y$")
plt.xlabel("$p(y)$")
plt.gca().xaxis.set_label_position("top")

#%%

gs2 = gs[2].subgridspec(
    nrows   = 2,
    ncols   = 2,
    hspace  = 0.,
    wspace  = 0.,
    height_ratios   = [1,0.2],
    width_ratios    = [0.2,1.0])


Xg,Yg = np.meshgrid(
    np.linspace(xlims[0],xlims[1],251),
    np.linspace(ylims[0],ylims[1],251))

Zg = np.zeros(Xg.shape)

for col in range(251):
    
    Zg[:,col] = np.exp(logprior(Xg[0,col]) + scipy.stats.norm.logpdf(
        x       = Yg[:,col],
        loc     = funpoly(Xg[0,col]),
        scale   = 1.25))





plt.subplot(gs2[0,1])

plt.plot(x,funpoly(x),ls='--',color="#666666",alpha=0.25)

plt.contour(
    Xg,
    Yg,
    Zg,
    cmap = cmap)


plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.gca().set_xlim(xlims)
plt.gca().set_ylim(ylims)


plt.title("$\mathbf{C}:$ Joint probability distribution",loc="left")


# Plot the function line
plt.plot(
    [0.75,0.75],
    [ylims[0],8.2],
    color = "xkcd:cobalt",
    zorder = -1,
    ls = ":")

plt.fill(
    [0.,0.75,0.75,0],
    [4.18,4.18,8.2,8.2],
    facecolor = "xkcd:cobalt",
    alpha = 0.25,
    zorder = -1,
    edgecolor = "None")

plt.text(
    0.775,
    0,
    "$x^{*}$",
    ha = "left",
    va = "center",
    color = "xkcd:cobalt",
    fontsize=10)

plt.text(
    0.4,
    funpoly(0.75),
    "$p(y|x^{*})$",
    ha = "center",
    va = "center",
    color = "xkcd:cobalt",
    fontsize=10)







plt.subplot(gs2[1,1])



y_prior = prior(x)

plt.fill(
    [0]+list(x)+[1],
    [0]+list(y_prior)+[0],
    facecolor = hcolor)


plt.gca().set_xlim(xlims)

plt.gca().invert_yaxis()
plt.gca().set_ylim([plt.gca().get_ylim()[0]*1.1,0])

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.xlabel("marginal random variable $x$")
plt.ylabel("$p(x)$")


plt.subplot(gs2[0,0])


y_prior = prior(x)

# plt.fill(
#     [0]+list(xpts)+[0],
#     [ypts[0]]+list(ypts)+[ypts[-1]],
#     facecolor = vcolor)

plt.fill(
    [0]+list(np.sum(Zg,axis=-1))+[0],
    [ypts[0]]+list(Yg[:,0])+[ypts[-1]],
    facecolor = vcolor)

# plt.gca().set_xlim(xlims)

plt.gca().invert_xaxis()


plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.gca().set_ylim(ylims)
plt.gca().set_xlim([plt.gca().get_xlim()[0]*1.1,0])

plt.ylabel("marginal random variable $y$")
plt.xlabel("$p(y)$")
plt.gca().xaxis.set_label_position("top")


plt.savefig('deterministic_to_stochastic.png',dpi=600,bbox_inches='tight')
plt.savefig('deterministic_to_stochastic.pdf',dpi=600,bbox_inches='tight')

raise Exception


#%%


gs2 = gs[0].subgridspec(
    nrows   = 2,
    ncols   = 2,
    hspace  = 0.,
    wspace  = 0.,
    height_ratios   = [1,0.2],
    width_ratios    = [0.2,1.0])

plt.subplot(gs2[0,0])

plt.fill(y,x,color=vcolor)
plt.gca().invert_xaxis()

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.ylim(ylims[0],ylims[1])
plt.xlim(plt.gca().get_xlim()[0],0)


plt.gca().xaxis.set_label_position('top') 
plt.ylabel("prior $p(a)$",color=vcolor,fontsize=14)



plt.ylim([0,1])

# Calculate the joint density
resolution  = 101
X,Y     = np.meshgrid(
    np.linspace(xlims[0],xlims[1],resolution),
    np.linspace(ylims[0],ylims[1],resolution))
Y       = np.flipud(Y)
Z       = np.zeros(X.shape)

# Independent coupling
for row in range(resolution):
    Z[row,:]    = prior_pdf(Y[row,0])
for row in range(resolution):
    # Z[:,col]    *= scipy.stats.norm.pdf(np.linspace(0,1,resolution),loc=X[0,col]**2,scale=0.075)
    Z[row,:]    *= scipy.stats.norm.pdf(X[row,:],loc=fun(Y[row,0]),scale=0.075)
    
    
    
    
plt.subplot(gs2[0,1])
plt.contour(X,Y,Z,cmap='Greys')
plt.plot(fun(x),x,ls='--')

def hslice(y):

    y1  = scipy.stats.norm.pdf(
        loc     = fun(y),
        scale   = 0.075,
        x       = np.linspace(-0.75,0.75,resolution))
    y1  /= np.max(y1)
    y1  *= 0.1
    plt.fill(
        np.linspace(-0.75,.75,resolution),
        y1+y,
        alpha   = 0.5,
        zorder  = 10,
        edgecolor = "None",
        color   = hcolor)
    plt.plot([-.75,.75],[y,y],color=vcolor,zorder = 100)
    
    return

hslice(0.2)
# hslice(0.625)
hslice(0.7)
# hslice(0.775)

plt.xlim(xlims[0],xlims[1])
plt.ylim(ylims[0],ylims[1])

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.text(
    x   = 0.45,
    y   = 0.75,
    s   = "likelihood $p(b|a)$",
    transform = plt.gca().transAxes,
    ha = "left",
    va = "center",
    fontsize = 14,
    color = hcolor)

plt.text(
    x   = 0.5,
    y   = 0.25,
    s   = "likelihood $p(b|a)$",
    transform = plt.gca().transAxes,
    ha = "right",
    va = "center",
    fontsize = 14,
    color = hcolor)


plt.text(
    x   = 0.7,
    y   = 0.5,
    s   = "$p(a,b)$",
    transform = plt.gca().transAxes,
    ha = "center",
    va = "center",
    fontsize = 14,
    color = "xkcd:grey")

# plt.title("$\mathbf{joint}$ $\mathbf{distribution}$")

plt.title("joint distribution $p(a,b)$", fontsize = 14)




plt.subplot(gs2[1,1])

x2  = np.linspace(-0.75,0.75,resolution)
y2  = np.sum(Z,axis=0)

x2  = np.concatenate(([-0.75],x2,[0.75]))
y2  = np.concatenate(([0],y2,[0]))

plt.fill(x2,y2,color=hcolor)

plt.xlim(-0.75,0.75)

plt.ylim(plt.gca().get_ylim()[-1],0)

plt.gca().set_xticks([])
plt.gca().set_yticks([])

# plt.text(
#     x   = 0.1,
#     y   = 0.25,
#     s   = "$p(b)$",
#     transform = plt.gca().transAxes,
#     ha = "center",
#     va = "center",
#     fontsize = 14,
#     color = hcolor)


plt.xlabel("marginal $p(b)$",color=hcolor,fontsize=14)
# plt.gca().yaxis.set_label_position('right') 
# plt.ylabel("$p(b)$",color=hcolor,fontsize=14)







#%%


gs3 = gs[1].subgridspec(
    nrows   = 2,
    ncols   = 2,
    hspace  = 0.,
    wspace  = 0.,
    height_ratios   = [1,0.2],
    width_ratios    = [0.2,1.0])

plt.subplot(gs3[0,0])

plt.fill(y,x,color=vcolor)
plt.gca().invert_xaxis()

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.ylim(0,1)
plt.xlim(plt.gca().get_xlim()[0],0)


plt.ylabel("prior $p(a)$",color=vcolor,fontsize=14)

# plt.gca().annotate('', xy=(-0.5,0.5), xycoords='axes fraction', xytext=(-3.25,0.5), 
#             arrowprops=dict(arrowstyle = '->',color='xkcd:grey',lw=2.))

# plt.text(
#     x   = -1.75,
#     y   = 0.5,
#     s   = "conditioning \n $p(a,b)$",
#     transform = plt.gca().transAxes,
#     ha = "center",
#     va = "center",
#     fontsize = 14)

plt.ylim([0,1])
    
plt.subplot(gs3[0,1])
plt.contour(X,Y,Z,cmap='Greys')
plt.plot(fun(x),x,ls='--')

def vslice(idx):

    y       = copy.copy(Z[:,idx])
    y       /= np.max(y)
    y       *= 0.2
    print(y)
    plt.fill(
        X[:,idx]+y,
        Y[:,idx],
        alpha   = 0.5,
        zorder  = 10,
        edgecolor = "None",
        color   = vcolor)
    plt.plot([X[0,idx],X[-1,idx]],[0,1],color=hcolor,zorder = 100)
    
    return

vslice(54)

plt.xlim(-0.75,0.75)
plt.ylim(0.,1.)

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.text(
    x   = 0.2,
    y   = 0.55,
    s   = "$p(a,b)$",
    transform = plt.gca().transAxes,
    ha = "center",
    va = "center",
    fontsize = 14,
    color = "xkcd:grey")

plt.text(
    x   = 0.56,
    y   = 0.05,
    s   = "observation $b^{*}$",
    transform = plt.gca().transAxes,
    ha = "left",
    va = "center",
    fontsize = 14,
    color = hcolor)

plt.text(
    x   = 0.575,
    y   = 0.575,
    s   = "posterior $p(a|b^{*})$",
    transform = plt.gca().transAxes,
    ha = "left",
    va = "center",
    fontsize = 14,
    color = vcolor)

plt.title("joint distribution $p(a,b)$", fontsize = 14)




plt.subplot(gs3[1,1])

x2  = np.linspace(-0.75,0.75,resolution)
y2  = np.sum(Z,axis=0)

x2  = np.concatenate(([-0.75],x2,[0.75]))
y2  = np.concatenate(([0],y2,[0]))

plt.fill(x2,y2,color=hcolor)

plt.xlim(-0.75,0.75)

plt.ylim(plt.gca().get_ylim()[-1],0)

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.xlabel("marginal $p(b)$",color=hcolor,fontsize=14)

# plt.text(
#     x   = 0.1,
#     y   = 0.25,
#     s   = "$p(b)$",
#     transform = plt.gca().transAxes,
#     ha = "center",
#     va = "center",
#     fontsize = 14,
#     color = hcolor)

plt.savefig('bayes_theorem.png',dpi=600,bbox_inches='tight')
plt.savefig('bayes_theorem.pdf',dpi=600,bbox_inches='tight')