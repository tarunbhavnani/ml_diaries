#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:30:50 2018

@author: tarun.bhavnani@dev.smecorner.com
"""

import os
os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/py-codes/ml-course/mlcourse.ai-master/data'
)
import pandas as pd
import numpy as np
df = pd.read_csv('telecom_churn.csv')
df.head()
print(df.shape)
print(df.columns)
print(df.info)

df['Churn']
df['Churn'].astype('int64')

df['Churn']= df['Churn'].astype('int64')

df.describe()


#In order to see statistics on non-numerical features, one has to explicitly indicate 
#data types of interest in the include parameter.

df.describe(include=['object', 'bool'])

#For categorical (type object) and boolean (type bool) features we can use the value_counts method. Let’s have a look at the distribution of Churn:

df['Churn'].value_counts()


#To calculate the proportion, pass normalize=True to the value_counts function

df['Churn'].value_counts(normalize=True)




#sorting
df.sort_values(by='Total day charge', ascending=False).head()

#sorting by multiple cols
df.sort_values(by=['Churn', 'Total day charge'], ascending=[True, False]).head()


"Indexing"

df['Churn'].mean()


"Boolean Indexing  df[P(df['Name'])]"

df[df['Churn']==1].mean()

#will give mean of all when churn=1

"""How much time (on average) do churned users spend on phone during daytime?

so this is how we get diff imp values, but what to look for has to be decieded manually"""


df[df['Churn'] == 1]['Total day minutes'].mean()


#What is the maximum length of international calls among loyal users (Churn == 0) who do not have an international plan?

df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max()



"Loc and ILOC"

#The loc method is used for indexing by name, while iloc() is used for indexing by number.

df.loc[0:5, 'State':'Area code']


df.iloc[0:5, 0:3]


"Applying Functions to Cells, Columns and Rows"
#To apply functions to each column, use apply():

df.apply(np.max)


"Apply function in each line"
#The apply method can also be used to apply a function to each line. To do this, specify axis=1


#For example, if we need to select all states starting with W
df[df['State'].apply(lambda state: state[0] == 'W')].head()




"Map, The map method can be used to replace values in a column by passing "
"a dictionary of the form {old_value: new_value} as its argument:"

d = {'No' : False, 'Yes' : True} 
df['International plan'] = df['International plan'].map(d) 
df['International plan'].head()


#Same thing can be done with the replace method:

df['Voice mail plan']
df = df.replace({'Voice mail plan': d}) 
df['Voice mail plan']



"Grouping"

#df.groupby(by=grouping_columns)[columns_to_show].function()

"""First, the groupby method divides the grouping_columns by their values. They become a new 
   index in the resulting dataframe.

   Then, columns of interest are selected (columns_to_show). If columns_to_show is not 
   included, all non groupby clauses will be included.

   Finally, one or several functions are applied to the obtained groups per selected columns."""
   
   
columns_to_show = ['Total day minutes', 'Total eve minutes', 
                   'Total night minutes']

df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])
#View in a dataframe

df_groupby_churn= df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])




#Summary Tables
#contingency tables using crosstab


pd.crosstab(df['Churn'], df['International plan'])

#proportion
pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True)


#Pivot Tables

#the pivot_table method takes the following parameters:

#values - a list of variables to calculate statistics for,
#index – a list of variables to group data by,
#aggfunc — what statistics we need to calculate for groups - e.g sum, mean, maximum, minimum or something else.
#Let’s take a look at the average numbers of day, evening and night calls by area code:

df_pivot_area=df.pivot_table(values=['Total day calls', 'Total eve calls', 'Total night calls'],
               index=['Area code'], aggfunc='mean')




#Unique Values

set(df['Area code'])



#transforming

#add coloumns

df['Total calls']
total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls'] 
df.insert(loc=len(df.columns), column='Total calls', value=total_calls) 
df.head()

#or
df['Total charge'] = df['Total day charge'] + df['Total eve charge'] +  df['Total night charge'] + df['Total intl charge']



#Delete Columns
"""To delete columns or rows, use the drop method, passing the required indexes and the 
axis parameter (1 if you delete columns, and nothing or 0 if you delete rows). The inplace 
argument tells whether to change the original DataFrame. With inplace=False, the drop method 
doesn't change the existing DataFrame and returns a new one with dropped rows or columns. 
With inplace=True, it alters the DataFrame."""

# get rid of just created columns 
df.drop(['Total charge', 'Total calls'], axis=1, inplace=True) 
df['Total charge']

# and here’s how you can delete rows 
df.head()

df.drop([1, 2]).head()
#look at index




"Predicting"
pd.crosstab(df['Churn'], df['International plan'])
pd.crosstab(df['Churn'], df['International plan'], margins=True)
pd.crosstab(df['Churn'], df['International plan'], margins=True, normalize=True)

# some imports and "magic" commands to set up plotting 
%matplotlib inline
import matplotlib.pyplot as plt 
# pip install seaborn 
import seaborn as sns
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='International plan', hue='Churn', data=df)


#Next, let’s look at another important feature — Customer service calls. Let’s also make a summary table and a picture.

pd.crosstab(df['Churn'], df['Customer service calls'], margins=True)

sns.countplot(x='Customer service calls', hue='Churn', data=df);

#lets put a binary attribute  calls > 3

df['Many_service_calls'] = (df['Customer service calls'] > 3).astype('int')
pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True, normalize=True)
sns.countplot(x='Many_service_calls', hue='Churn', data=df);

#Let’s construct another contingency table that relates Churn with both International plan and freshly created Many_service_calls.

pd.crosstab(df['Many_service_calls'] & df['International plan'] , df['Churn'])





#Table
#value_counts

df['Many_service_calls'].value_counts()









































