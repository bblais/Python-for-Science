"""Generic object pickler and compressor

This module saves and reloads compressed representations of generic Python
objects to and from the disk.
"""

__author__ = "Bill McNeill <billmcn@speakeasy.net>"
__version__ = "1.0"

import pickle
import gzip
import os
from numpy import array 
from copy import deepcopy

def Save(object, filename='_memory_.dat'):
    """Saves an object to disk
    
    Example:  Save([1,2,3])
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, 1))
    file.close()


def Load(filename='_memory_.dat'):
    """Loads an object from disk

    Example:  a=Load()
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while 1:
        data = file.read()
        if data == "":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object



def Remember(*args,**kwargs):

    try:
        filename=kwargs['filename']
    except KeyError:
        filename='_memory_.dat'

    if len(args)>0:
        Save(args,filename)
        return

    Q=Load(filename)
    if len(Q)==1:
        Q=Q[0]
        
    return Q
    
__memory_data={}
__memory_data['default']=[]
def reset(name=None):
    global __memory_data
    
    if name==None:
        __memory_data={}
        __memory_data['default']=[]
    else:
        __memory_data[name]=[]
    
def store(*args,**kwargs):
    global __memory_data
    
    if 'name' in kwargs:
        name=kwargs['name']
    else:
        name='default'
    
    if name not in __memory_data:
        __memory_data[name]=[]
        
    if not args:
        __memory_data[name]=[]
    
    if not __memory_data[name]:
        for arg in args:
            __memory_data[name].append([deepcopy(arg)])
            
    else:
        for d,a in zip(__memory_data[name],args):
            d.append(deepcopy(a))
    

def recall(name='default'):
    global __memory_data
    
    if name not in __memory_data:
        __memory_data[name]=[]
    
    tmp=[]
    for i in range(len(__memory_data[name])):
        tmp.append(array(__memory_data[name][i]))
        
    
    ret=tuple(tmp)
    if len(ret)==1:
        return ret[0]
    else:
        return ret




if __name__ == "__main__":
    import sys
    import os.path
    
    class Object:
        x = 7
        y = "This is an object."
    
    filename = sys.argv[1]
    if os.path.isfile(filename):
        o = load(filename)
        print "Loaded %s" % o
    else:
        o = Object()
        save(o, filename)
        print "Saved %s" % o

