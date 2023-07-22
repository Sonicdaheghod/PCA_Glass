#!/usr/bin/env python
# coding: utf-8

# In[28]:


#Youtube video code referred to : https://youtu.be/yDUCqI4zBlM?t=520

#dataset used: German,B.. (1987). Glass Identification. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P.


# # Principal Component Analysis (PCA) 
# 
# ![image.png](attachment:image.png)
# #### Image source: https://www.keboola.com/blog/pca-machine-learning
# 
# ## This is used to:
# 
# ### - reduce the dimensionality of large data sets. In other words, if there are multiple factors describing a single aspect (ex. people have age, weight, height, gender, etc), 
# 
# ### - determine the factor influencing the correlation between independent and dependent variables

# In[29]:


# pip install pydataset


# ## 1) Import Modules

# In[30]:


import pandas as pd
from sklearn.decomposition import PCA
from pydataset import data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[31]:


#glass dataset info

# 1. Id number: 1 to 214
# 2. RI: refractive index
# 3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 4. Mg: Magnesium
# 5. Al: Aluminum
# 6. Si: Silicon
# 7. K: Potassium
# 8. Ca: Calcium
# 9. Ba: Barium
# 10. Fe: Iron
# 11. Type of glass: (class attribute)
#      -- 1 building_windows_float_processed
#      -- 2 building_windows_non_float_processed
#      -- 3 vehicle_windows_float_processed
#      -- 4 vehicle_windows_non_float_processed (none in this database)
#      -- 5 containers
#      -- 6 tableware
#      -- 7 headlamps


# ## 2) Reading Dataset

# In[32]:


import pandas as pd
# reading csv files, prints first 3 items
glass_data =  pd.read_csv('glass.data', sep=",")


pandas_glass = pd.DataFrame(glass_data)
print(pandas_glass)


# In[33]:


#add appropriate columns
glass_data.columns = ["Id","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type of glass"]

print(pandas_glass)


# ## 3) Standardize data

# In[34]:


scaler = StandardScaler()
glass_data_scaled = scaler.fit_transform(glass_data) #changes to training data to it can be used for PCA
# glass_data_scaled = pd.Dataframe(glass_data_scaled)

#scales to mean = 0, 1 = Standard Deviation

glass_data_scaled = pd.DataFrame(glass_data_scaled)

glass_data_scaled


# ### 3a) Label top column on stardardized data table

# In[35]:


glass_data_scaled = glass_data_scaled.rename(index=str, columns={0:"Id number",1:"Refractive_index",2:"Sodium",3:"Magnesium",4:"Aluminum",5:"Silicon",6:"Potassium",7:"Calcium",8:"Barium",9:"Iron",10:"Type of glass"})

glass_data_scaled


# ## 4) PCA Analysis

# In[36]:


#a - we want two components to work with 
pca_2c=PCA(n_components = 2)
#b - fit.transform() allows turning data into something our PCA model can work with 
X_pca_2c=pca_2c.fit_transform(glass_data_scaled)
#c- print out array fo data our model will use
X_pca_2c


# In[37]:


#this will tell us what is causing the variance in our dataset

pca_2c.explained_variance_ratio_

#component 1 has a 33% effect on the variance of our dataset and component 2 has about 20% effect on the variance of the dataset.


# In[38]:


#summing the variance percentage of both components
pca_2c.explained_variance_ratio_.sum()


# ## Plotting PCA data on graph

# In[ ]:


##this plots data under a varaible of our choice and turns that into PCA (one variable broken down into two features and PCA determines variability)


# In[39]:


plt.scatter(X_pca_2c[:,0],X_pca_2c[:,1],c=glass_data_scaled.Refractive_index)


# In[ ]:




