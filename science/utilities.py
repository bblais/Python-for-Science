import time

from Struct import *
from Memory import *
from Waitbar import Waitbar

def HTML(string):
    try:
        from IPython.core.display import publish_html
        publish_html(string)
    except ImportError:
        from IPython.core.display import display_html
        display_html(string,raw=True)

def time2str(tm):
    
    frac=tm-int(tm)
    tm=int(tm)
    
    s=''
    sc=tm % 60
    tm=tm//60
    
    mn=tm % 60
    tm=tm//60
    
    hr=tm % 24
    tm=tm//24
    dy=tm

    if (dy>0):
        s=s+"%d d, " % dy

    if (hr>0):
        s=s+"%d h, " % hr

    if (mn>0):
        s=s+"%d m, " % mn


    s=s+"%.2f s" % (sc+frac)

    return s

def timeit(reset=False):
    global _timeit_data
    try:
        _timeit_data
    except NameError:
        _timeit_data=time.time()
        
    if reset:
        _timeit_data=time.time()

    else:
        return time2str(time.time()-_timeit_data)


import gc
import timeit as timeit_orig

class Timer:
    def __init__(self, timer=None, disable_gc=False, verbose=True):
        if timer is None:
            timer = timeit_orig.default_timer
        self.timer = timer
        self.disable_gc = disable_gc
        self.verbose = verbose
        self.start = self.end = self.interval = None
    def __enter__(self):
        if self.disable_gc:
            self.gc_state = gc.isenabled()
            gc.disable()
        self.start = self.timer()
        return self
    def __exit__(self, *args):
        self.end = self.timer()
        if self.disable_gc and self.gc_state:
            gc.enable()
        self.interval = self.end - self.start
        if self.verbose:
            print('time taken: %s' % time2str(self.interval))
