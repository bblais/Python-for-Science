import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas

import os,sys
from scipy.stats.stats import nanmean
from scipy.stats import distributions as D
from numpy import *
from pylab import *
import pylab as py
from utilities import time,time2str,Struct,timeit
from plot_utilities import histogram,figure
from copy import deepcopy

greek=['alpha','beta','gamma','chi','tau','sigma','lambda',
        'epsilon','zeta','xi','theta','rho','psi','mu','nu']

def remove_nan(x,y):
    try:
        x=x[y.notnull()]
        y=y[y.notnull()]
    except AttributeError:
        x=x[~isnan(y)]
        y=y[~isnan(y)]
        
    return x,y
    
    
    
def fit(x,y,funcstr,*args,**kwargs):

    x=pandas.Series(array(x))
    y=pandas.Series(array(y))

    x,y=remove_nan(x,y)
    
    
    if funcstr=='linear':
        result=fit(x,y,'power',1)
        result.type='linear'
    elif funcstr=='quadratic':
        result=fit(x,y,'power',2)
        result.type='quadratic'
    elif funcstr=='exponential':
        y2=np.log(y)
        result=fit(x,y2,'linear')
        result.params=[np.exp(result.params[1]),result.params[0]]
        p=result.params
        labelstr='y= %.4e exp(%.4e x)' % (p[0],p[1])
        result.label=labelstr
        result.type='exponential'
    
    elif funcstr=='power':
        data=pandas.DataFrame({'x':x,'y':y})
        power=args[0]
        
        keys=['x']
        for i in range(power-1):
            exponent=(i+2)
            key='x%d' % exponent
            data[key] = x**exponent
            keys.append(key)

        result2=pandas.ols(y=data['y'],x=data[keys])
        keys.reverse()
        keys+=['intercept']
        
        p=[result2.beta[s] for s in keys]

        labelstr='y= '
        for i,pv in enumerate(p):
            pw=len(p)-i-1
            if pw==1:
                labelstr+='%.4e x + ' % (pv)
            elif pw==0:
                labelstr+='%.4e + ' % (pv)
            else:
                labelstr+='%.4e x^%d + ' % (pv,pw)
        labelstr=labelstr[:-3]  # take off the last +
        
        
        result=Struct()
        result.params=p
        result.type='power'
        result.label=labelstr   
        result.pandas_result=result2
        
    else:
        raise ValueError,'Unknown fit name %s' % funcstr
        
    return result
        
def fitval(result,x):
    x=pandas.Series(array(x))

    if result.type=='linear':
        y=result.params[0]*x+result.params[1]
    elif result.type=='quadratic':
        y=result.params[0]*x**2+result.params[1]*x+result.params[2]
    elif result.type=='power':
        y=0.0
        for i,pv in enumerate(result.params):
            pw=len(result.params)-i-1
            y+=pv*x**pw
    elif result.type=='exponential':
        y=result.params[0]*np.exp(x*result.params[1])
    else:
        raise ValueError,'Unknown fit name %s' % result.type
        
    return y
    
      

try:
    import emcee
    import triangle
except ImportError:
    pass
    


def normal(x,mu,sigma):
    return 1/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/2.0/sigma**2)

def num2str(a):
    from numpy import abs
    if a==0:
        sa=''
    elif 0.001<abs(a)<10000:
        sa='%g' % a
    else:
        sa='%.3e' % a
        parts=sa.split('e')

        parts[1]=parts[1].replace('+00','')
        parts[1]=parts[1].replace('+','')
        parts[1]=parts[1].replace('-0','-')
        parts[1]=parts[1].replace('-0','')
        
        sa=parts[0]+r'\cdot 10^{%s}'%parts[1]
    
    return sa
    

    
def linear_equation_string(a,b):
    
    astr=num2str(a)
    bstr=num2str(abs(b))
    
    if b<0:
        s=r'$y=%s\cdot x - %s$' % (astr,bstr)
    else:
        s=r'$y=%s\cdot x + %s$' % (astr,bstr)
    
    return s
    
def quadratic_equation_string(a,b,c):
    
    astr=num2str(a)
    bstr=num2str(abs(b))
    cstr=num2str(abs(c))
    
    s=r'$y=%s\cdot x^{2}' % astr
    
    
    if b<0:
        s+=r' - %s\cdot x' % (bstr)
    else:
        s+=r' - %s\cdot x' % (bstr)
    
    if c<0:
        s+=r' - %s$' % (cstr)
    else:
        s+=r' - %s$' % (cstr)

    return s

from scipy.special import gammaln,gamma
def logfact(N):
    return gammaln(N+1)

def lognchoosek(N,k):
    return gammaln(N+1)-gammaln(k+1)-gammaln((N-k)+1)

def loguniformpdf(x,mn,mx):
    if mn < x < mx:
        return np.log(1.0/(mx-mn))
    return -np.inf

def logjeffreyspdf(x):
    if x>0.0:
        return -np.log(x)
    return -np.inf

def lognormalpdf(x,mn,sig):
    # 1/sqrt(2*pi*sigma^2)*exp(-x^2/2/sigma^2)
    try:
        N=len(x)
    except TypeError:
        N=1
        
    return -0.5*np.log(2*np.pi*sig**2)*N - np.sum((x-mn)**2/sig**2/2.0)
    
def logbernoullipdf(theta, h, N):
    return lognchoosek(N,h)+np.log(theta)*h+np.log(1-theta)*(N-h)

def logbetapdf(theta, h, N):
    return logfact(N+1)-logfact(h)-logfact(N-h)+np.log(theta)*h+np.log(1-theta)*(N-h)

def logexponpdf(x,_lambda):
    # p(x)=l exp(l x)
    return _lambda*x + np.log(_lambda)

import scipy.optimize as op

class Normal(object):
    def __init__(self,mean=0,std=1):
        self.mean=mean
        self.std=std
        self.default=mean
        
    def rand(self,*args):
        return np.random.randn(*args)*self.std+self.mean
    
    def __call__(self,x):
        return lognormalpdf(x,self.mean,self.std)
class Exponential(object):
    def __init__(self,_lambda=1):
        self._lambda=_lambda

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return logexponpdf(x,self._lambda)


class Uniform(object):
    def __init__(self,min=0,max=1):
        self.min=min
        self.max=max
        self.default=(min+max)/2.0
       
    def rand(self,*args):
        return np.random.rand(*args)*(self.max-self.min)+self.min
        
    def __call__(self,x):
        return loguniformpdf(x,self.min,self.max)

class Jeffries(object):
    def __init__(self):
        self.default=1.0
        
    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return logjeffreyspdf(x)

class Beta(object):
    def __init__(self,h=100,N=100):
        self.h=h
        self.N=N
        self.default=float(h)/N

    def rand(self,*args):
        return np.random.rand(*args)
        
    def __call__(self,x):
        return logbetapdf(x,self.h,self.N)

class Bernoulli(object):
    def __init__(self,h=100,N=100):
        self.h=h
        self.N=N
        self.default=float(h)/N

    def rand(self,*args):
        return np.random.rand(*args)
        
    def __call__(self,x):
        return logbernoullipdf(x,self.h,self.N)
     

def lnprior_function(model):
    def _lnprior(x):
        return model.lnprior(x)

    return _lnprior

class MCMCModel_Meta(object):

    def __init__(self,**kwargs):
        self.params=kwargs
        
        self.keys=[]
        for key in self.params:
            self.keys.append(key)


        self.index={}
        for i,key in enumerate(self.keys):
            self.index[key]=i


        self.nwalkers=100
        self.burn_percentage=0.25
        self.initial_value=None
        self.samples=None
        self.last_pos=None

    def lnprior(self,theta):
        pass

    def lnlike(self,theta):
        pass

    def lnprob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)        

    def __call__(self,theta):
        return self.lnprob(theta)

    def set_initial_values(self,method='prior',*args,**kwargs):
        if method=='prior':
            ndim=len(self.params)
            try:
                N=args[0]
            except IndexError:
                N=300

            pos=zeros((self.nwalkers,ndim))
            for i,key in enumerate(self.keys):
                pos[:,i]=self.params[key].rand(100)

            
            self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, 
                    lnprior_function(self))

            timeit(reset=True)
            print("Sampling Prior...")
            self.sampler.run_mcmc(pos, N,**kwargs)
            print("Done.")
            print timeit()

            # assign the median back into the simulation values
            self.burn()
            self.median_values=np.percentile(self.samples,50,axis=0)

            self.last_pos=self.sampler.chain[:,-1,:]
        elif method=='samples':
            lower,upper=np.percentile(self.samples, [16,84],axis=0)            
            subsamples=self.samples[((self.samples>=lower) & (self.samples<=upper)).all(axis=1),:]
            idx=np.random.randint(subsamples.shape[0],size=self.last_pos.shape[0])
            self.last_pos=subsamples[idx,:]            


        elif method=='maximum likelihood':
            self.set_initial_values()
            chi2 = lambda *args: -2 * self.lnlike_lownoise(*args)
            result = op.minimize(chi2, self.initial_value)
            vals=result['x']
            self.last_pos=emcee.utils.sample_ball(vals, 
                            0.05*vals+1e-4, size=self.nwalkers)
        elif method=='median':            
            vals=self.median_values
            self.last_pos=emcee.utils.sample_ball(vals, 
                            0.05*vals+1e-4, size=self.nwalkers)
        else:
            raise ValueError,"Unknown method: %s" % method

    def burn(self,burn_percentage=None):
        if not burn_percentage is None:
            self.burn_percentage=burn_percentage
            
        burnin = int(self.sampler.chain.shape[1]*self.burn_percentage)  # burn 25 percent
        ndim=len(self.params)
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    def run_mcmc(self,N,**kwargs):
        ndim=len(self.params)
        
        if self.last_pos is None:
            self.set_initial_values()
        
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self,)
        
        timeit(reset=True)
        print("Running MCMC...")
        self.sampler.run_mcmc(self.last_pos, N,**kwargs)
        print("Done.")
        print timeit()

        # assign the median back into the simulation values
        self.burn()
        self.median_values=np.percentile(self.samples,50,axis=0)
        theta=self.median_values

        self.last_pos=self.sampler.chain[:,-1,:]


    def plot_chains(self,*args,**kwargs):
        py.clf()
        
        if not args:
            args=self.keys
        
        
        fig, axes = py.subplots(len(self.params), 1, sharex=True, figsize=(8, 5*len(args)))
        try:  # is it iterable?
            axes[0]
        except TypeError:
            axes=[axes]



        labels=[]
        for ax,key in zip(axes,args):
            i=self.index[key]
            sample=self.sampler.chain[:, :, i].T

            if key.startswith('_sigma'):
                label=r'$\sigma$'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='$%s$' % namestr

            labels.append(label)
            ax.plot(sample, color="k", alpha=0.2,**kwargs)
            ax.set_ylabel(label)

    def triangle_plot(self,*args,**kwargs):
        
        if not args:
            args=self.keys
            
        assert len(args)>1
        
        labels=[]
        idx=[]
        for key in args:
            if key.startswith('_sigma'):
                label=r'$\sigma$'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='$%s$' % namestr

            labels.append(label)
            idx.append(self.index[key])
        
        fig = triangle.corner(self.samples[:,idx], labels=labels,**kwargs)

            
    def plot_distributions(self,*args,**kwargs):
        if not args:
            args=self.keys
        
        for key in args:
            if key.startswith('_sigma'):
                label=r'\sigma'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='%s' % namestr

            i=self.index[key]
            
            py.figure(figsize=(12,4))
            result=histogram(self.samples[:,i],bins=200)
            xlim=py.gca().get_xlim()
            x=py.linspace(xlim[0],xlim[1],500)
            y=D.norm.pdf(x,np.median(self.samples[:,i]),np.std(self.samples[:,i]))
            py.plot(x,y,'-')

            v=np.percentile(self.samples[:,i], [2.5, 50, 97.5],axis=0)
            py.title(r'$\hat{%s}^{97.5}_{2.5}=%.3f^{+%.3f}_{-%.3f}$' % (label,v[1],(v[2]-v[1]),(v[1]-v[0])))
            py.ylabel(r'$p(%s|{\rm data})$' % label)
            py.xlabel(r'$%s$' % label)
                
    def get_distribution(self,key,bins=200):
            
        i=self.index[key]
        x,y=histogram(self.samples[:,i],bins=bins,plot=False)
        
        return x,y
        
    def percentiles(self,p=[16, 50, 84]):
        result={}
        for i,key in enumerate(self.keys):
            result[key]=np.percentile(self.samples[:,i], p,axis=0)
            
        return result
        
    def best_estimates(self):
        self.median_values=np.percentile(self.samples,50,axis=0)
        theta=self.median_values
        
        return self.percentiles()
    
    def get_samples(self,*args):
        result=[]
        for arg in args:
            idx=self.keys.index(arg)
            result.append(self.samples[:,idx])
    
        return result
    
 

class MCMCModel2(MCMCModel_Meta):
    def __init__(self,data,lnprior,lnlike,**kwargs):

        self.data=data
        self.lnprior_function=lnprior
        self.lnlike_function=lnlike

        MCMCModel_Meta.__init__(self,**kwargs)

    def lnprior(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            params_dict[key]=theta[i]
                
        return self.lnprior_function(**params_dict)

    def lnlike(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            params_dict[key]=theta[i]
                
        return self.lnlike_function(self.data,**params_dict)

    


class MCMCModel(MCMCModel_Meta):
    
    def __init__(self,x,y,function,**kwargs):
        self.x=x
        self.y=y
        self.function=function
        self.params=kwargs
        
        MCMCModel_Meta.__init__(self,**kwargs)

        self.params['_sigma']=Jeffries()
        self.keys.append('_sigma')        
        self.index['_sigma']=len(self.keys)-1


    # Define the probability function as likelihood * prior.
    def lnprior(self,theta):
        value=0.0
        for i,key in enumerate(self.keys):
            value+=self.params[key](theta[i])
                
        return value

    def lnlike(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            if key=='_sigma':
                sigma=theta[i]
            else:
                params_dict[key]=theta[i]
                
        y_fit=self.function(self.x,**params_dict)
        
        return lognormalpdf(self.y,y_fit,sigma)
    
    def lnlike_lownoise(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            if key=='_sigma':
                sigma=1.0
            else:
                params_dict[key]=theta[i]
                
        y_fit=self.function(self.x,**params_dict)
        
        return lognormalpdf(self.y,y_fit,sigma)


    def predict(self,x,theta=None):
        
        if theta is None:
            self.percentiles()
            theta=self.median_values
            
        args={}
        for v,key in zip(theta,self.keys):
            if key=='_sigma':
                continue
            args[key]=v

        y_predict=array([self.function(_,**args) for _ in x])        
        return y_predict
    
    def plot_predictions(self,x,N=1000):
        samples=self.samples[-N:,:]
        
        for value in samples:
            args={}
            for v,key in zip(value,self.keys):
                if key=='_sigma':
                    continue
                args[key]=v

            y_predict=array([self.function(_,**args) for _ in x])        
            plot(x,y_predict,color="k",alpha=0.05)
