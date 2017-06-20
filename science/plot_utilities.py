import matplotlib.pyplot as plt
import pylab
from pylab import *
from numpy import *
numpyhist=histogram

def histogram(y,bins=50,plot=True,label=None):
    N,bins=numpyhist(y,bins)
    
    dx=bins[1]-bins[0]
    if dx==0.0:  #  all in 1 bin!
        val=bins[0]
        bins=plt.linspace(val-abs(val),val+abs(val),50)
        N,bins=numpyhist(y,bins)
    
    dx=bins[1]-bins[0]
    x=bins[0:-1]+(bins[1]-bins[0])/2.0
    
    y=N*1.0/sum(N)/dx
    
    if plot:
        plt.plot(x,y,'o-',label=label)
        yl=plt.gca().get_ylim()
        plt.gca().set_ylim([0,yl[1]])
        xl=plt.gca().get_xlim()
        if xl[0]<=0 and xl[0]>=0:    
            plt.plot([0,0],[0,yl[1]],'k--')

    return x,y


def despine():
    ax=plt.gca()
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    