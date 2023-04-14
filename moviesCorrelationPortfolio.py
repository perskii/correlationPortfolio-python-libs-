#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')
#sets the default size of Matplotlib figures to 12 inches wide by 8 inches tall.
matplotlib.rcParams['figure.figsize']=(12,8)
 
# read in the data from csv file 
df = pd.read_csv(r'C:\Users\hmllb\Downloads\movies.csv')

# delete all missing data (null data) from a DataFrame. 
# The dropna() method removes any row that contains at least one null value by default.
df = df.dropna()

#change display settings (show all rows)
pd.set_option('display.max_rows', None)



# In[3]:


# basic look at the data

df.head()


# In[4]:


# way to check if there is any missing data (null value)
# 1. 
for col in df.columns:
    missing_data = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,missing_data))
print('\n')
# 2.
for col in df.columns:
    print(df[col].isnull().value_counts(), " \n------------------ \n")
# 3.
df.isnull().sum() 


# In[12]:


# other way to check if there is any missing data (NaN value)
# nan = null
any_missing = df.isna().any()
missing_rows_company = df.loc[df['company'].isna()]
missing_rows_count = df['company'].isna().sum()
missing_rows_count
print("---------------------------------------------")
missing_rows_company




# In[6]:


#data types for columns
print(df.dtypes)


# In[7]:


#change dtype of columns

df['budget']= df['budget'].astype('int64')
df['gross']= df['gross'].astype('int64')
df['votes']= df['votes'].astype('int64')
df['runtime']= df['runtime'].astype('int64')

print(df.dtypes)


# In[8]:


#create correct year release (based on column "release")
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)


# In[9]:


#sorting data by value in "gross" column
df.sort_values(by=['gross'], inplace = False, ascending = False).head()


# In[10]:


# Are there any Outliers?

df.boxplot(column=['gross'])


# In[11]:


# Order data a little bit to see
df.sort_values(['company'],ascending = False)



# In[ ]:


#drop any duplicates if exist 
df['company']= df['company'].drop_duplicates()


# In[13]:


#budget high correlation
#company high correlation
#scatter plot with budget vs gross

plt.scatter(
    x=df['budget'], 
    y= df['gross'],
    color='#fcba03')

plt.title('budget vs gross earinings')
plt.xlabel('budget for film ')
plt.ylabel('gross earinigns')
plt.show()


# In[14]:


#plot => budget vs gross using seaborn

sns.regplot(
    x='budget',
    y='gross',
    data= df, 
    scatter_kws ={"color": "orange"}, 
    line_kws={"color": "blue"})


# In[15]:


sns.regplot(
    x="score", 
    y="gross", 
    data=df,
    scatter_kws ={"color": "orange"}, 
    line_kws={"color": "blue"})


# In[16]:


#looking at correlation between all numeric columns

print("pearson method")

df.corr(numeric_only = True,method='pearson')


# In[17]:


print("kendall method")

df.corr(numeric_only = True, method='kendall')


# In[18]:


print("spearman method")

df.corr(numeric_only = True,method='spearman')


# In[19]:


correlation_matrix = df.corr(numeric_only = True, method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('correlation matric for numeric features')
plt.xlabel('Movie features')
plt.ylabel('Movie features')

plt.show()


# In[20]:


# Using factorize - this assigns a random numeric value 
# for each unique categorical value
df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[21]:


correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('correlation matric for numeric features')
plt.xlabel('Movie features')
plt.ylabel('Movie features')

plt.show()


# In[22]:


correlation_mat = df.apply(lambda x: x.factorize()[0]).corr()
corr_pair = correlation_mat.unstack()
print(corr_pair)


# In[23]:


sorted_pairs= corr_pair.sort_values(kind="quicksort")
print(sorted_pairs)


# In[85]:


# look at the ones that have a high correlation (> 0.5)
#summary: Votes and budget have the highest correlation to gross (earnings)

high_corr = sorted_pairs[(sorted_pairs) > 0.5]

high_corr


# In[24]:


# Looking at the top 15 compaies by gross revenue

CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted



# In[25]:


# Final result: a DataFrame containing the top 15 company names, years, and their corresponding total gross sums,
# sorted in descending order based on gross sums, company names, and years
CompanyGrossSum = df.groupby(['company', 'year'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company','year'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[26]:


#looks at company
# string type
df_numerized = df.copy()

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.add_categories(['missing'])
        df_numerized[col_name] = df_numerized[col_name].cat.codes
df_numerized.head()


# In[37]:


sns.stripplot(x="rating", y="gross", data=df,palette="Set2")


# In[ ]:




