
# coding: utf-8

# In[1]:

from science import *


# In[2]:

temperature_data=pandas.read_csv('temperatures.txt',sep='\s*')  # cleaned version
sunspot_data=pandas.read_csv('spot_num.txt',sep='\s*') 


# In[7]:

x=temperature_data['Year']
y=temperature_data['J-D']/100.0

plot(x,y,'-o')
ylabel('Temperature Anomaly')
xlabel('Year')


# In[12]:

x=sunspot_data['YEAR']+(sunspot_data['MON']-1)/12.0
y=sunspot_data['SSN']
plot(x,y,'-o')
xlabel('year')
ylabel('sunspot number')


# In[18]:




# In[37]:

ax1=gca()
x=temperature_data['Year']
y=temperature_data['J-D']/100.0
plot(x,y,'-o')
ylabel('Temperature')

x=sunspot_data['YEAR']+(sunspot_data['MON']-1)/12.0
y=sunspot_data['SSN']
y2=pandas.rolling_mean(y,150)

ax2 = gca().twinx()
plot(x,y2,'r-')
ylabel('Sunspot Number')
ax2.set_ylim([20,120])
ax1.set_xlim([1880,2013])
ax2.set_xlim([1880,2013])


# In[34]:

get_ipython().magic(u'pinfo pandas.rolling_mean')


# In[ ]:



