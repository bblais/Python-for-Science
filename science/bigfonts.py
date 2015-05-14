def bigfonts(size=20,family='sans-serif'):
    
    from matplotlib import rc
    
    rc('font',size=size,family=family)
    rc('axes',labelsize=size)
    rc('axes',titlesize=size)
    rc('xtick',labelsize=size)
    rc('ytick',labelsize=size)
    rc('legend',fontsize=size)

bigfonts()