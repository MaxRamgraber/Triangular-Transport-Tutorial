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


def prior_pdf(x):
    
    pdf     = np.zeros(x.shape)
    
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

plt.fill(
    [0]+list(np.sum(Zg,axis=-1))+[0],
    [ypts[0]]+list(Yg[:,0])+[ypts[-1]],
    facecolor = vcolor)

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