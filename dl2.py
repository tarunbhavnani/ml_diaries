#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:52:41 2019

@author: tarun.bhavnani
"""

##############3
#create classificatio data

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

Y=pd.get_dummies(y)


len_out=9

X1=df


#X,Y are for normal classification data
#X1,Y are when we have put some non linearity in data


#we will create more data

#circles, this might help us in seeing how a next hidden layer is helpful

from sklearn.datasets import make_circles
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()




X, y = make_circles(n_samples=10000, noise=0.05)
X1= PCA(n_components=2).fit_transform(X)
Y= pd.get_dummies(y)

from keras.models  import Sequential
from keras.layers import Dense

#one layer, logistic regression 
model= Sequential()
model.add(Dense(units= 2, input_shape=(X.shape[1],), activation="sigmoid"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

history=model.fit(X,Y,epochs=50, batch_size=32, validation_split=.3)


#two layer, it basically cuts the circles in two dimensions and cut a hyperplane
model= Sequential()
model.add(Dense(units= 5, input_shape=(X.shape[1],), activation="relu"))
model.add(Dense(units= 2,  activation="sigmoid"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

history=model.fit(X,Y,epochs=50, batch_size=32, validation_split=.3)


model= Sequential()
model.add(Dense(units= 20, input_shape=(X.shape[1],), activation="relu"))
model.add(Dense(units= 10, input_shape=(X.shape[1],), activation="relu"))

model.add(Dense(units= 2,  activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

history=model.fit(X,Y,epochs=50, batch_size=32, validation_split=.3)



#tsne took a very long time as compared to pca!!




