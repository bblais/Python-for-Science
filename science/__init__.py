import os
import matplotlib.pyplot as plt
import Struct

path,junk=os.path.split(Struct.__file__)
plt.style.use(path+'/science.mplstyle')


from utilities import *
from plot_utilities import *
from stat_utilities import *



__version__="0.0.3"
