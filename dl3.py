#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:58:03 2019

@author: tarun.bhavnani
"""

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import pandas as pd
import numpy as np
# generate 2d classification dataset
X, y = make_blobs(n_samples=10000, centers=9, n_features=5)

df=pd.DataFrame(data=X, columns=("x1","x2","x3","x4","x5"))

df["x1"]=abs(df["x1"])**(1/3)
df["x2"]=abs(df["x2"])**(1/2)

df= round(10*df)

Y=pd.get_dummies(y)

asd=np.corrcoef(df["x1"],df["x5"])

#check all corrs

import itertools

#[i for i in itertools.combinations([1,2,3],2)]

for i,j in itertools.combinations(list(df),2):
    #print(i,j)
    a= np.corrcoef(df[i],df[j])
#    print(i,j,"-----",a[0,1])
    if abs(a[0,1])>.01:
     print(i,j,"-----",a[0,1])
    else:
     print(i,j,"-----no correlation")

    

from sklearn.decomposition import PCA

pca= PCA(n_components=5)
pca.fit(df)
asd=pca.fit_transform(df)
#pca.components_
dfpca=pd.DataFrame(data=asd, columns=("px1","px2","px3","px4","px5"))


for i,j in itertools.combinations(list(dfpca),2):
    #print(i,j)
    a= np.corrcoef(dfpca[i],dfpca[j])
    if abs(a[0,1])>.01:
     print(i,j,"-----",a[0,1])
    else:
     print(i,j,"-----no correlation")






###
     
     ##




#are they similar!!??
a=[0,5,10,15,20,25,30,35,40,45,50,55,60]
b=[0,0,10,15,20,25,30,35,40,45,50,55,60]
c=[0,0,5,15,2,25,30,35,40,5,50,5,6]

ttest,pval = scipy.stats.ttest_rel(c,b)
if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

ttest,pval = scipy.stats.ttest_rel(a,c)
pval
if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")




####
  "WHICH LOSS TO USE!!  "
    
# generate regression dataset
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)    


from sklearn.preprocessing import StandardScaler
# standardize dataset
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
 

# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# define model
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='...', optimizer=opt)
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)












