
# coding: utf-8

# ## Trying to fit a poisson process

# In[1]:

get_ipython().magic(u'matplotlib inline')
from science import *
from scipy.stats import poisson


# In[2]:

x=np.random.poisson(lam=10,size=1000)


# In[3]:

def lnprior(mu):
    if 0<=mu<=100:
        return 0.0
    return -np.inf


# In[4]:

def lnlike(data,mu):
    return log(mu)*sum(data)-mu*len(data)


# In[5]:

model=MCMCModel2(x,lnprior,lnlike,
                mu=Uniform(0,100))


# In[6]:

model.run_mcmc(500)
model.plot_chains()


# In[7]:

model.plot_distributions()


# ## What if my data were 2 or 3 data points?

# In[21]:

x=np.random.poisson(lam=10,size=1)
print x


# In[22]:

model=MCMCModel2(x,lnprior,lnlike,
                mu=Uniform(0,100))


# In[23]:

model.run_mcmc(500)
model.plot_chains()


# In[24]:

model.plot_distributions()


# In[ ]:



