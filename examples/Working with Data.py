
# coding: utf-8

# In[1]:

from science import *


# ## Some Data Sets
# 
# 1. [Surface Temperature Data - http://data.giss.nasa.gov/gistemp/](http://data.giss.nasa.gov/gistemp/)
# 2. [Solar Spot Number Data](http://solarscience.msfc.nasa.gov/SunspotCycle.shtml)
# 

# ## Global Surface Temperature
# 
# Global surface temperature data from [http://data.giss.nasa.gov/gistemp/](http://data.giss.nasa.gov/gistemp/).

# In[2]:

data=pandas.read_csv('temperatures.txt',sep='\s*')  # cleaned version


# In[3]:

data


# In[4]:

plot(data['Year'],data['J-D'],'-o')
xlabel('Year')
ylabel('Temperature Deviation')


# Actually, the deviation is 1/100 of this, so let's adjust...

# In[5]:

x=data['Year']
y=data['J-D']/100.0
plot(x,y,'-o')
xlabel('Year')
ylabel('Temperature Deviation')


# ## Or if you like Excel

# In[6]:

xls = pandas.ExcelFile('temperatures.xls')

print xls.sheet_names

data=xls.parse('Sheet 1')
data


# ## Station Data
# 
# This data is from [http://data.giss.nasa.gov/gistemp/station_data/](http://data.giss.nasa.gov/gistemp/station_data/)

# In[7]:

data=pandas.read_csv('station.txt',sep='\s*')
data


# This plot will look weird, because of the 999's. 

# In[8]:

x,y=data['YEAR'],data['metANN']
plot(x,y,'-o')
xlabel('Year')
ylabel('Temperature Deviation')


# replace the 999's with Not-a-Number (NaN) which is ignored in plots.

# In[9]:

y[y>400]=NaN


# In[10]:

plot(x,y,'-o')
xlabel('Year')
ylabel('Temperature Deviation')


# ## Fitting
# 
# ### First, ordinary least squares (ols)

# In[11]:

model=pandas.ols(x=x,y=y)
print model.summary
print "Beta",model.beta


# In[12]:

m,b=model.beta['x'],model.beta['intercept']
plot(x,y,'-o')
x1=linspace(1890,2000,100)
y1=x1*m+b
plot(x1,y1,'-')
xlabel('Year')
ylabel('Temperature Deviation')


# In[13]:

data


# ### Next, try fitting a polynomial

# In[14]:

result=fit(x,y,'power',2)


# In[15]:

xfit = linspace(1850,2000,100)
yfit = fitval(result,xfit)
plot(x,y,'-o')
plot(xfit,yfit,'-')
xlabel('Year')
ylabel('Temperature Deviation')


# printing out the results of the fit.

# In[16]:

result


# This should do the same thing.

# In[17]:

result=fit(x,y,'quadratic')


# In[18]:

result


# ## Do a super-crazy high polynomial

# In[19]:

result=fit(x,y,'power',4)
xfit = linspace(1890,1980,100)
yfit = fitval(result,xfit)
plot(x,y,'-o')
plot(xfit,yfit,'-')
xlabel('Year')
ylabel('Temperature Deviation')


# In[20]:

result


# In[20]:



