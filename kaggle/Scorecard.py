# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:27:00 2023

@author: ELECTROBOT
Viquar hashmi and Tracy .
"""

import os
import pandas as pd
os.chdir(r'D:\kaggle\scorecards')


rejected=pd.read_csv(r"D:\kaggle\scorecards\rejected_2007_to_2018Q4.csv")

hd=rejected.sample(frac=1).head(2000)

null_r=pd.DataFrame(rejected.isnull().sum())

accepted= pd.read_csv(r"D:\kaggle\scorecards\accepted_2007_to_2018Q4.csv")
hd1= accepted.sample(frac=1).head(2000)
list(hd1)
hd1.settlement_status.value_counts()
null_a=pd.DataFrame(accepted.isnull().sum())
null_a=null_a.reset_index()
null_a.columns= ["index", "miss"]
null_a["pc"]=null_a['miss'].divide(len(accepted))*100


describe_a=accepted.describe()


corr= accepted.corr()
corr=corr.fillna(0)
high_corr={}
for col in corr.columns:
    for row in corr.index:
        
        value= corr.loc[row,col]
        if value>.5 and value<1 and (col, row) not in high_corr:
            high_corr[(row,col)]=value


accepted.loan_status.value_counts()


hd1.loan_status
# =============================================================================
# 
# =============================================================================

#missing values
null_a=pd.DataFrame(accepted.isnull().sum())
null_a=null_a.reset_index()
null_a.columns= ["index", "miss"]
null_a["pc"]=null_a['miss'].divide(len(accepted))*100

var=null_a[null_a['pc']<.80]['index']

accepted= accepted[var]
accepted=accepted[~accepted.loan_amnt.isnull()]
#accepted=accepted[~accepted.revol_util.isnull()]

#test just remove all which have any nulls
# in reality go to each variable check what makes most sense and replace 
for i in list(accepted):
    accepted=accepted[~accepted[i].isnull()]

#but data is big so we can live with just removing all for testing purpose!





accepted['status']=[0 if i in ['Fully Paid','Current'] else 1 for i in accepted['loan_status']]

accepted['status'].value_counts(normalize=True)

accepted['grade'].value_counts()

#univariate analysis
import seaborn as sns
import matplotlib.pyplot as plt
def plot_cat(cat_var):
    sns.barplot(x='cat_var', y='status', data=accepted)
    plt.show()



plt.figure(figsize=(16, 6))
plot_cat('sub_grade')

#see how the status fairs across variables

sns.distplot(accepted['loan_amnt'])

#bin the laons and check the default across

loan_map={0:50000, 1:100000,2:200000,3:300000, 4:400000, 5:500000}

accepted['loan_amnt_bins']=[5 if i<=50000 else 999 for i in accepted['loan_amnt']]
accepted['loan_amnt_bins']=[4 if i<=40000 else j for i,j in zip(accepted['loan_amnt'],accepted['loan_amnt_bins'])]
accepted['loan_amnt_bins']=[3 if i<=30000 else j for i,j in zip(accepted['loan_amnt'],accepted['loan_amnt_bins'])]
accepted['loan_amnt_bins']=[2 if i<=20000 else j for i,j in zip(accepted['loan_amnt'],accepted['loan_amnt_bins'])]
accepted['loan_amnt_bins']=[1 if i<=10000 else j for i,j in zip(accepted['loan_amnt'],accepted['loan_amnt_bins'])]
accepted['loan_amnt_bins']=[0 if i<=5000 else j for i,j in zip(accepted['loan_amnt'],accepted['loan_amnt_bins'])]

accepted['loan_amnt_bins'].value_counts()

plot_cat('loan_amnt_bins')




plt.figure(figsize=(16, 6))
plt.xticks(rotation=40)
plot_cat('purpose')

plt.figure(figsize=(16, 6))
plt.xticks(rotation=40)
plot_cat('term')

#multivariate
def plot_segmented(cat_var):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_var, y='status', hue='purpose', data=accepted)
    plt.show()

    


plot_segmented('term')




# =============================================================================
# woe
# =============================================================================

import numpy as np
feature,target = 'loan_amnt_bins','status'
df_woe_iv = (pd.crosstab(accepted[feature],accepted[target],
                      normalize='columns')
             .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0])))
            


gh=accepted.groupby('loan_amnt_bins').agg({"status":'sum', 'term':'count'})
gh.columns=['bad', 'total']

gh['good']= gh['total']-gh['bad']
gh['good']= gh['good']/sum(gh['good'])
gh['bad']= gh['bad']/sum(gh['bad'])


gh['woe']= [np.log(i/j) for i,j in zip(gh.good, gh.bad)]

iv= sum([(i-j)*k for i,j,k in zip(gh.good, gh.bad, gh.woe)])







