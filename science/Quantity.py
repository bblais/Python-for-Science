import pandas
import numpy as np

class Quantity(object):
    def __init__(self,Ns=50000,fix=None):
        self.Ns=Ns    # number of samples
        self.orig_Ns=self.Ns
        self._fix_values=False
        
      
    @property
    def fix_values(self):
        return self._fix_values
    
    @fix_values.setter
    def fix_values(self, value):
        self._fix_values=value
        
        if self._fix_values:
            self.orig_Ns=self.Ns
            
            self.samples=self.value.reshape(1,self.Nd)
        else:
            self.Ns=self.orig_Ns
            self.generate_samples()
        
    @property
    def ndim(self):
        return 2
        
    @property
    def mean(self):
        return self.samples.mean(axis=0)

    @property
    def std(self):
        return self.samples.std(axis=0)

    @property
    def shape(self):
        return self.samples.shape
    
    
    @property
    def median(self):
        return np.median(self.samples,axis=0)
        
    @property
    def percentile95(self):
        return np.percentile(self.samples,[2.5,97.5],axis=0)

    def percentile(self,p=[2.5,50,97.5]):
        return np.percentile(self.samples,p,axis=0)
    
    
    def error(self,errorbar_level=68):
        ylmu=self.percentile([  (100-errorbar_level)/2,
                                50,
                                100-(100-errorbar_level)/2,
                             ])
        
        L,M,U=ylmu[0,:],ylmu[1,:],ylmu[2,:]

        yerr=np.array([M-L,U-M])

        return yerr
    
    
    def __len__(self):
        return len(self.samples)
    
    def __add__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
            
        new=Quantity()
        new.samples=self.samples+samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new
    def __sub__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=self.samples-samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

    def __radd__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=samples+self.samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new
    def __rsub__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=samples-self.samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

    def __mul__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=self.samples*samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

    def __rmul__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=samples*self.samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

    def __neg__(self):
        
        new=Quantity()
        new.samples=-self.samples.copy()
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

    def __abs__(self):
        
        new=Quantity()
        new.samples=np.abs(self.samples)
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

    def __pos__(self):
        
        new=Quantity()
        new.samples=self.samples.copy()
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

    def __pow__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=self.samples**samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new
        
    
    def __rtruediv__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=samples/self.samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new
        
    
    def __truediv__(self,other):
        if isinstance(other,Quantity):
            samples=other.samples
        else:
            samples=np.atleast_2d(other).T
        
        new=Quantity()
        new.samples=self.samples/samples
        new.Ns,new.Nd=new.samples.shape
        self.value=self.mean
    
        return new

        
    def __iter__(self):
        return self.samples.__iter__()
    
    def __getslice__(self, i, j):
        return self.samples[i:j,:]
    
    def __getitem__(self,key):
        return self.samples[key]
    
    def __repr__(self):
        if self.Nd==1:
            s=str(self.mean)+" +- "+str(self.std)
            s+="\n"
            pp=self.percentile95.ravel()
            s+=str(self.median) + " :: 95% range\t["+str(pp[0])+" - "+str(pp[1])+"]"
            return s
        else:
            pp=self.percentile()
            s="95% range (\n"
            for l,m,u in pp.T:
                s+="\t"+str(m) + " ::\t["+str(l)+" - "+str(u)+"],\n"
            s+=")\n"
            return s
        

class Quniform(Quantity):
    
    def __init__(self,lower,upper,value=None,**kwargs):
        super().__init__(**kwargs)
        
        try:
            self.Nd=len(lower)
            assert len(lower)==len(upper)
            
            lower=np.array(lower)
            upper=np.array(upper)
        except TypeError:
            self.Nd=1
            
        self.lower=lower
        self.upper=upper
        if value is None:
            self.value=np.array((upper+lower)/2)
        else:
            self.value=np.array(value)

        self.generate_samples()
        
        if 'fix' in kwargs:
            self.fix_values=kwargs['fix']
        
    def generate_samples(self):
        self.samples=np.random.rand(self.Ns,self.Nd)*(self.upper-self.lower)+self.lower
        
class Qnormal(Quantity):
    
    def __init__(self,mean,std,**kwargs):
        super().__init__(**kwargs)
        
        try:
            self.Nd=len(mean)
            mean=np.array(mean)
            std=np.array(std)
        except TypeError:
            self.Nd=1
            
        self._std=std
        self._mean=mean
        self.value=np.array(mean)
        self.generate_samples()

        if 'fix' in kwargs:
            self.fix_values=kwargs['fix']
        
    def generate_samples(self):
        self.samples=np.random.randn(self.Ns,self.Nd)*self._std+self._mean
        
def log(x):
    if isinstance(x,Quantity):
        y= +x
        y.samples=np.log(y.samples)
    else:
        y=np.log(x)
        
    return y

def exp(x):
    if isinstance(x,Quantity):
        y= +x
        y.samples=np.exp(y.samples)
    else:
        y=np.exp(x)
        
    return y
        
def make_uncertainties_dataframe(errorbar_level=68,**kwargs):
    df=pandas.DataFrame()
    Nd=None

    for key in kwargs:
        var=kwargs[key]
        if isinstance(var,Quantity):
            if Nd is None:
                Nd=var.Nd
                if Nd==1:
                    print("First variable to save should be a full-length variable not a constant.")

            ylmu=var.percentile([  (100-errorbar_level)/2,
                                    50,
                                    100-(100-errorbar_level)/2,
                                ])

            if var.Nd==Nd:
                df[key]=ylmu[1]
                df[key+"_L"]=ylmu[0]
                df[key+"_U"]=ylmu[2]
            elif var.Nd==1:
                df[key]=ylmu[1]*np.ones(Nd)
                df[key+"_L"]=ylmu[0]*np.ones(Nd)
                df[key+"_U"]=ylmu[2]*np.ones(Nd)
            else:
                raise ValueError("%s has the wrong size %s" % (key,var.shape))

        elif isinstance(var,float) or isinstance(var,int):
            df[key]=var*np.ones(Nd)
            df[key+"_L"]=np.nan*np.ones(Nd)
            df[key+"_U"]=np.nan*np.ones(Nd)
        else:
            df[key]=var
            df[key+"_L"]=np.nan*np.ones(len(var))
            df[key+"_U"]=np.nan*np.ones(len(var))

        
    return df

