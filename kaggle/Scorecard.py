# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:27:00 2023

@author: ELECTROBOT
"""

import os
import pandas as pd
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\scorecards')


rejected=pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\scorecards\rejected_2007_to_2018Q4.csv")


hd=rejected.sample(frac=1).head(2000)
null_r=pd.DataFrame(rejected.isnull().sum())

accepted= pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\scorecards\accepted_2007_to_2018Q4.csv")
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
