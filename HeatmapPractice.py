#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# In[29]:


flights_data = sns.load_dataset("flights")


# In[30]:


flights_data


# In[31]:


flights_data.describe()


# In[14]:


flights = flights_data.pivot(index="year", columns="month", values="passengers")
plt.figure(figsize =(10,7))

ax = sns.heatmap(flights)
plt.title("Heatmap of flight data")
plt.show()


# In[20]:


flights = flights_data.pivot(index="year", columns="month", values="passengers")
plt.figure(figsize =(10,7))

ax = sns.heatmap(flights, linewidths = 0.5, annot = True, fmt ='d')
plt.title("Heatmap of flight data")
plt.show()

#compare result with the describe function like min, max


# In[32]:


from sklearn.datasets import fetch_california_housing #get previously saved data
housing_data = fetch_california_housing()


# In[34]:


df = pd.DataFrame(housing_data.data) #convert into pandas dataframe
df.columns = housing_data.feature_names
df['Price'] = housing_data.target
df.head()


# In[35]:


df.corr() #correlation between columns


# In[43]:


plt.figure(figsize=(10,7))
ax = sns.heatmap(df.corr(),annot = True)
plt.title("California housing data")
plt.show()


# In[44]:


plt.figure(figsize=(10,7))
ax = sns.heatmap(df.corr(),annot = True)
plt.title("California housing data")
plt.show()


# In[42]:


#plotting the lower part, use the mask function

import numpy as np

# Create a mask for the upper part
mask = np.triu(np.ones_like(df.corr())) #np.tril for the upper part

# Plot the heatmap with the mask
plt.figure(figsize=(10,7))
ax = sns.heatmap(df.corr(), annot=True, mask=mask, fmt = "0.3f")
plt.title("California Housing Data")
plt.show()


# In[ ]:




