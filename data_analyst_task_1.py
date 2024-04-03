#!/usr/bin/env python
# coding: utf-8

# # Data Acquisition and Understanding
# 

# In[ ]:


### Importing the libraries that will be needed



import numpy as np
import pandas as pd
from matploylib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')



# In[1]:


### importing of the dataset that will be used
import pandas as pd


df = pd.read_csv(r"C:\Users\NtomfuthiNtshangase\Desktop\ Data-Analyst-Task-1\vgsales.csv")

print(df)


# In[12]:


# Decribing the dataset including the mean, std, min, max. and percentage of the database
import pandas as pd
df.describe()


# # 2.Data Cleaning and Preparation:

# In[11]:


### Cleaning the dataset by checking duplicate on the data given

import pandas as pd
df.duplicated().sum()


# In[13]:


### cleaning the dataset by checking missing information

import pandas as pd

missing_info = df.isnull().sum()
print(missing_info)


# In[14]:


### Cleaning  the dataset by Removing the outliers 
 
import pandas as pd
from scipy import stats

z_scores = stats.zscore(df['NA_Sales'])
threshold=3
df_no_outliers=df[abs(z_scores)<threshold]

print(df_no_outliers)


# In[10]:


### Cleaning the dataset by formating the data

import pandas as pd

df['Year'] = pd.to_datetime(df['Year'], format='%Y')
print(df)


# #  3.Descriptive Analysis 

# In[9]:


### Visualize data distributions using histograms, box plots, or density plots

### Displaying the dataset using Boxplot for the NA_sales

import seaborn as sns
sns.set(style="whitegrid")
ax=(sns.boxplot(x=df.NA_Sales,showfliers=False))


# In[7]:


### Ploting histogram for the dataset
import pandas as pd
import seaborn as sns
sns.histplot(df['Year'].dropna(), kde=False, bins=39)


# In[33]:


### Density plot 

import pandas as pd
import matplotlib.pyplot as plt

sns.histplot(df['Year'], kde=True, bins=int(180/5), color='darkblue', edgecolor='black', linewidth=1.5)
plt.show()




# In[8]:


### Perform summary statistics

import pandas as pd

summary_stats=df.describe()

print("Summary Statistics:")
print(summary_stats)


# In[16]:


### Identifying the mean of the datasets

import pandas as pd
median_values = df.mean(numeric_only=True)

print("\nMean Values:")
print(mean_values)


# In[17]:


### Identifying the median of the dataset

import pandas as pd
median_values = df.median(numeric_only=True)

print("\nMedian Values:")
print(median_values)


# In[18]:


### Identifying the mode of the dataset

import pandas as pd
mode_values = df.mode().iloc[0]

print("\nMode Values:")
print(mode_values)


# In[19]:


### Identifying the standard deviation for the business metrics

import pandas as pd
std_values = df.std(numeric_only=True)

print("\nStandard Deviation Values:")
print(std_values)


# # 4.Segmentation and Profiling

# In[3]:


import pandas as pd

high_sales = df[df['Global_Sales'] > df['Global_Sales'].quantile(0.75)]
medium_sales = df[(df['Global_Sales'] <= df['Global_Sales'].quantile(0.75)) & (df['Global_Sales'] > df['Global_Sales'].quantile(0.25))]
low_sales = df[df['Global_Sales'] <= df['Global_Sales'].quantile(0.25)]


high_sales_profile = high_sales.describe()
medium_sales_profile = medium_sales.describe()
low_sales_profile = low_sales.describe()


print("High Sales Segment Profile:")
print(high_sales_profile)

print("\nMedium Sales Segment Profile:")
print(medium_sales_profile)

print("\nLow Sales Segment Profile:")
print(low_sales_profile)


# In[5]:


### Create customer or product profiles to understand their characteristics and behaviors.


import pandas as pd

profile =df.describe(include='all')

print("Profile:")
print(profile)






# #  5.Correlation and Trends
# 

# In[22]:


### Analyzing correlation between different business metrics

import pandas as pd


selected_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
selected_data = df[selected_columns]


# calculation of the correlation
correlation_matrix = selected_data.corr(numeric_only=True)



print("Correlation Matrix:")
print(correlation_matrix)


# In[36]:


# analyzing the sale marketing decisions
import pandas as pd


sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

correlation_matrix = df[sales_columns].corr()

print("Correlation Matrix:")
print(correlation_matrix)


# In[ ]:





# In[ ]:




