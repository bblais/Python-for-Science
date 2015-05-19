
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from science import *


# ## Make up some data

# In[2]:

data=randn(1000)+40


# In[3]:

hist(data)


# ## Make it x-y data

# In[4]:

x=arange(len(data))
y=data
plot(x,y,'o')


# # Constant Model
# ## Define Model

# In[5]:

def constant(x,a):
    return a

model=MCMCModel(x,y,constant,
            a=Uniform(0,100),
            )
model.run_mcmc(500)
model.plot_chains()


# In[6]:

model.plot_distributions()


# In[7]:

model.triangle_plot()


# ## Replot the data with the fit

# In[8]:

plot(x,y,'o')

xfit=linspace(0,1000,200)
yfit=model.predict(xfit)

plot(xfit,yfit,'-')


# # Linear Model
# ## Define Model

# In[9]:

def linear(x,a,b):
    return a*x+b

model=MCMCModel(x,y,linear,
                a=Uniform(-10,10),
                b=Uniform(0,100),
                )

model.run_mcmc(500)
model.plot_chains()


# chains look weird?  run for some more...

# In[10]:

model.run_mcmc(500)
model.plot_chains()


# In[11]:

model.plot_distributions()


# In[12]:

model.triangle_plot()


# ## Replot the data with the fit

# In[13]:

plot(x,y,'o')

xfit=linspace(0,1000,200)
yfit=model.predict(xfit)

plot(xfit,yfit,'-')


# # Using some real data

# In[14]:

xls = pandas.ExcelFile('temperatures.xls')

print xls.sheet_names

data=xls.parse('Sheet 1')
data


# In[15]:

x=data['Year']
y=data['J-D']/100.0
plot(x,y,'-o')
xlabel('Year')
ylabel('Temperature Deviation')


# In[16]:

def linear(x,m,b):
    return m*x+b

x=data['Year']-1880
y=data['J-D']/100.0

model=MCMCModel(x,y,linear,
                m=Uniform(-10,10),
                b=Uniform(-100,100),
                )

model.run_mcmc(500)
model.plot_chains()


# In[17]:

model.plot_distributions()


# In[18]:

model.triangle_plot()


# In[19]:

plot(x+1880,y,'-o')

xfit=linspace(1870,2020,200)
yfit=model.predict(xfit-1880)

plot(xfit,yfit,'-')


# ## Sampling

# In[20]:

plot(x,y,'o')

xfit=linspace(1850,2050,200)
model.plot_predictions(xfit-1880,100)


# ## Quadratic?

# In[21]:

def quadratic(x,a,b,c):
    return a*x**2 + b*x + c


#  make the data a bit more numerically stable

# In[22]:

x=data['Year']
y=data['J-D']/100.0

# make a little more palatable
x=x-mean(x)
y=y-mean(y)

model=MCMCModel(x,y,quadratic,
                a=Uniform(-10,10),
                b=Uniform(-10,10),
                c=Uniform(-100,100),
                )

model.run_mcmc(500)
model.plot_chains()


# In[23]:

model.plot_distributions()
model.triangle_plot()

figure()
plot(x,y,'o')

xfit=linspace(min(x)-50,max(x)+50,200)
model.plot_predictions(xfit,100)    


# what is I used the raw data?

# In[24]:

x=data['Year']
y=data['J-D']/100.0

model=MCMCModel(x,y,quadratic,
                a=Uniform(-10,10),
                b=Uniform(-1000,1000),
                c=Uniform(-2000,2000),
                )

model.run_mcmc(500)
model.plot_chains()


# In[25]:

model.run_mcmc(500)
model.plot_chains()


# In[26]:

model.plot_distributions()
model.triangle_plot()

figure()
plot(x,y,'o')

xfit=linspace(min(x)-50,max(x)+50,200)
model.plot_predictions(xfit,100)    


# ## Logistic Model
# 
# http://mathinsight.org/bacteria_growth_logistic_model
# 
# definition of function from http://en.wikipedia.org/wiki/Logistic_function:
# 
# \begin{equation}
# f(x)=\frac{L}{1+e^{-k(x-x_o)}}
# \end{equation}

# In[27]:

def logistic(x,L,k,xo):
    return L/(1.0+exp(-k*(x-xo)))


# In[28]:

datastr="""0	0	0.022
16	1	0.036
32	2	0.060
48	3	0.101
64	4	0.169
80	5	0.266
96	6	0.360
112	7	0.510
128	8	0.704
144	9	0.827
160	10	0.928
"""

x,y=zip(*[(float(line.split()[0]),float(line.split()[2])) for line in datastr.split('\n') if line])
plot(x,y,'o')


# In[29]:

model=MCMCModel(x,y,logistic,
                L=Uniform(0,5),
                k=Uniform(0,20),
                xo=Uniform(0,1000),
                )
model.run_mcmc(500)
model.plot_chains()


# In[30]:

model.run_mcmc(500)
model.plot_chains()


# In[31]:

model.plot_distributions()
model.triangle_plot()

figure()
plot(x,y,'o')

xfit=linspace(min(x),max(x)+50,200)
model.plot_predictions(xfit,100)    


# In[ ]:




# In[ ]:



