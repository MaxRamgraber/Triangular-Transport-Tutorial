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

def fun(x):
    
    xl  = x*2 - 1
    
    return 2*xl**3 - 1*xl

def prior_pdf(x):
    
    pdf     = np.zeros(x.shape)
    
    pdf     = scipy.stats.beta.pdf(a = 15, b = 7, x = x)
    pdf     += scipy.stats.beta.pdf(a = 3, b = 8, x = x)
    
    return pdf

plt.close("all")

plt.figure(figsize=(16,8))

gs  = GridSpec(
    nrows           = 1,
    ncols           = 2,
    wspace          = 0.25,
    width_ratios    = [1,1])

x   = np.linspace(0,1,101)
y   = prior_pdf(x)

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
    Z[row,:]    *= scipy.stats.norm.pdf(X[row,:],loc=fun(Y[row,0]),scale=0.075)
    
plt.subplot(gs2[0,1])
plt.contour(X,Y,Z,cmap=cmap)
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
        alpha   = 0.75,
        zorder  = 10,
        edgecolor = "None",
        color   = hcolor)
    plt.plot([-.75,.75],[y,y],color=vcolor,zorder = 100)
    
    return

hslice(0.2)
hslice(0.7)

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

plt.xlabel("marginal $p(b)$",color=hcolor,fontsize=14)

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

plt.ylim([0,1])
    
plt.subplot(gs3[0,1])
plt.contour(X,Y,Z,cmap=cmap)
plt.plot(fun(x),x,ls='--')

def vslice(idx):

    y       = copy.copy(Z[:,idx])
    y       /= np.max(y)
    y       *= 0.2
    print(y)
    plt.fill(
        X[:,idx]+y,
        Y[:,idx],
        alpha   = 0.75,
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

plt.savefig('bayes_theorem.png',dpi=600,bbox_inches='tight')
plt.savefig('bayes_theorem.pdf',dpi=600,bbox_inches='tight')