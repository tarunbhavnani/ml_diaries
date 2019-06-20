#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:49:42 2019

@author: tarun.bhavnani
identity functions and resnets

"""

import numpy as np
x=np.random.randint(0,100,100)
y=pd.get_dummies(x)
#x=x.reshape(100,1)

model= Sequential()
#model.add(Input(shape=(1,)))
model.add(Dense(128, activation="relu", input_dim=1))
model.add(Dense(128, activation="relu"))
model.add(Dense(len(set(y)), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()
hi=model.fit(x,y, epochs=1000)

#acc is 40%
#its an identity but not learning! lets yuse resnets here!!

dat=np.random.randint(0,100,100)
y1= pd.get_dummies(dat)
#dat.reshape((1,1))
#x= Input(shape=(1,1))
x = Dense(28, activation="relu", input_dim=1)
# this returns x + y.
from keras.layers import add
z = add([x, y])
out= Dense(len(set(dat)), activation="softmax")(z)

model= Model(x,out)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()
hi=model.fit(dat,y1, epochs=1000)










# Weights are given as [weights, biases], so we give
# the identity matrix for the weights and a vector of zeros for the biases
weights = [np.diag(np.ones(84)), np.zeros(84)]

model = Sequential([Dense(84, input_dim=84, weights=weights)])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, X, nb_epoch=10, batch_size=8, validation_split=0.3)


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import pandas as pd

X = np.array([[random.random() for r in range(84)] for i in range(1,100000)])
model = Sequential([Dense(84, input_dim=84)], name="layer1")
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, X, nb_epoch=100, batch_size=80, validation_split=0.3)

l_weights = np.round(model.layers[0].get_weights()[0],3)

print l_weights.argmax(axis=0)
print l_weights.max(axis=0)