#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:38:24 2019

@author: tarun.bhavnani
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
X_trainc, X_testc, y_trainc, y_testc = train_test_split(cancer.data, cancer.target, test_size=0.3, stratify=cancer.target, random_state=30)
cancerclf = LogisticRegression()
cancerclf.fit(X_trainc, y_trainc)
#print "Logreg score on cancer data set", cancerclf.score(X_testc, y_testc) # you can check the score if you want, which is not the main purpose. 


probac = cancerclf.predict_proba(X_testc)
probac[1:10] 

predict = cancerclf.predict(X_testc)
predict [1:10]

probability = probac[:,0]
prob_df = pd.DataFrame(probability)
prob_df.head(10) # this should match the probac 1st column 

prob_df['predict'] = np.where(prob_df[0]>=0.90, 1, 0)# create a new column
prob_df.head(10)

prob_df['predict'] = np.where(prob_df[0]>=0.97, 1, 0)
prob_df.head(10)

import numpy as np
prob_df['predict'] = np.where(prob_df[0]>=0.50, 1, 0)
len(prob_df[prob_df['predict']==1])

prob_df['predict'] = np.where(prob_df[0]>=0.97, 1, 0)
len(prob_df[prob_df['predict']==1])




