#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:20:23 2019

@author: tarun.bhavnani
"""

#prepare data

import numpy as np
import pandas as pd


df=pd.DataFrame(data=np.random.randint(100, size=(10000,5)), columns=("x1","x2","x3","x4","x5"))
df=pd.DataFrame(data=np.random.rand(10000,5), columns=("x1","x2","x3","x4","x5"))



df["target"]= .5*df["x1"]**5+.4*df["x2"]**4+.3*df["x3"]**3+.2*df["x2"]**2+.1*df["x1"]+.2

df["target"]=round(100*df["target"])


#now we have some random data and a target, can we now build models to predict the target??

#first we bin so that to get target in categories
min(df["target"])
max(df["target"])


bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
labels = [1,2,3,4,5,6,7,8,9]
df['binned'] = pd.cut(df['target'], bins=bins, labels=labels)
df["binned"].value_counts()
#print (df)

X=df.iloc[:,0:5]
Y=df["binned"]

#split in test and train
#bin Y

Y=pd.get_dummies(df["binned"])



len_out=len(set(df["binned"]))

#now we have a non linear relation of parameters with the targets thats is "binned". we will try algos now!!


#1)

from keras.models  import Sequential
from keras.layers import Dense

model= Sequential()
model.add(Dense(units= len_out, input_shape=(5,), activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

history=model.fit(X,Y,epochs=50, batch_size=32, validation_split=.3)
history=model.fit(X1,Y,epochs=50, batch_size=32, validation_split=.3)

history.history["val_acc"][-1]
#only 59%

"is it because our target variable was exponentially connected to parameters/non linear"
"what if we put a non linera layer"


model= Sequential()
model.add(Dense(units= 32, input_shape=(5,), activation="relu"))
model.add(Dense(units= len_out, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

history=model.fit(X,Y,epochs=50, batch_size=32, validation_split=.3)

history.history["val_acc"][-1]
#87 % impressive


##more non linear layers

model= Sequential()
model.add(Dense(units= 64, input_shape=(5,), activation="relu"))
model.add(Dense(units= 32, activation="relu"))

model.add(Dense(units= len_out, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

history=model.fit(X,Y,epochs=100, batch_size=32, validation_split=.3,callbacks=[early_stop])

history.history["val_acc"][-1]
#92 % at only 26 epochs!!! impressive


##more

##more non linear layers

model= Sequential()
model.add(Dense(units= 128, input_shape=(5,), activation="relu"))
model.add(Dense(units= 64, activation="relu"))
model.add(Dense(units= 64, activation="relu"))
#model.add(Dense(units= 32, activation="relu"))

model.add(Dense(units= len_out, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

history=model.fit(X,Y,epochs=100, batch_size=32, validation_split=.3,callbacks=[early_stop])

history.history["val_acc"][-1]
#91 % at only 19 epochs!!! impressive


#but 92 % is the max we have




"""
i think since our parameters were nonlinearly related to the target so dense layers help, bt why cant they get 100
pc accuracy when it is actually connected?
lets break into train and test and see the real etst accuracy here


"""

#lets do the last one with pca df

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['p1', 'p2',"p3","p4","p5"])

model= Sequential()
model.add(Dense(units= 128, input_shape=(5,), activation="relu"))
model.add(Dense(units= 64, activation="relu"))
model.add(Dense(units= 64, activation="relu"))
#model.add(Dense(units= 32, activation="relu"))

model.add(Dense(units= len_out, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam" , metrics=["acc"])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

history=model.fit(principalDf,Y,epochs=100, batch_size=32, validation_split=.3,callbacks=[early_stop])

history.history["val_acc"][-1]
#96 % at only 12 epochs!!! impressive





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

"""
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
"""































