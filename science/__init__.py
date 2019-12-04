import os
import matplotlib.pyplot as plt
import Struct

path,junk=os.path.split(Struct.__file__)
plt.style.use(path+'/science.mplstyle')


from .utilities import *
from .plot_utilities import *
from .stat_utilities import *

def subplot(*args,**kwargs):  # avoids deprication error
    import pylab as plt
    try:
        fig=plt.gcf()
        if args in fig._stored_axes:
            plt.sca(fig._stored_axes[args])
        else:
            plt.subplot(*args,**kwargs)
            fig._stored_axes[args]=plt.gca()
    except AttributeError:
            plt.subplot(*args,**kwargs)
            fig._stored_axes={}
            fig._stored_axes[args]=plt.gca()

    return plt.gca()


__version__="0.0.4"
print("Version",__version__)

