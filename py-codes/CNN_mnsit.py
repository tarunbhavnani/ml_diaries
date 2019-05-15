#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:54:32 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test)= mnist.load_data()

X_train.shape

import matplotlib.pyplot as plt
plt.imshow(X_train[0])

X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

#the last "1" tells the model that its  greyscale, "3" would have been there for a rgb

#one hot encode Y
#Y_train= to_categorical(Y)
Y_train= pd.get_dummies(Y_train)
#Y_test=pd.get_dummies(Y_test)
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
model= Sequential()
model.add(Conv2D(filters=6, kernel_size=2, input_shape=(28,28,1), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(filters=12, kernel_size=3, activation="relu", padding="same"))
#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(Y_train.shape[1], activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

history=model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=.33)

hist=history.history


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()



############################

model= Sequential()
model.add(Conv2D(filters=6, kernel_size=2, input_shape=(28,28,1), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(filters=12, kernel_size=3, activation="relu", padding="same"))
#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(Y_train.shape[1], activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

history=model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=.33)

hist=history.history
history.history['acc'][-1] 99.35
history.history['loss'][-1] .020

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

##



model= Sequential()
model.add(Convolution2D(filters=6, kernel_size=2, input_shape=(28,28,1), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Convolution2D(filters=12, kernel_size=3, activation="relu", padding="same"))
model.add(GlobalAveragePooling2D())
#model.add(Flatten())
model.add(Dense(Y_train.shape[1], activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

history1=model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=.33)

hist=history.history
history1.history['acc'][-1] 99.35
history1.history['loss'][-1] .020

plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.show()


plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.show()


y_prob = model.predict(X_test) 
y_classes = y_prob.argmax(axis=-1)
Y_test

from sklearn.metrics import confusion_matrix
asd=confusion_matrix(y_true=Y_test, y_pred=y_classes, labels=[0,1,2,3,4,5,6,7,8,9])
asd=pd.DataFrame(asd)
tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=y_classes).ravel()
from collections import Counter
Counter(y_classes)
Counter(Y_test)

asd.sum(axis=0)#7:1005: vertical--predicted
asd.sum(axis=1)#7:1028: horizontal--real


#####Predicted
#
#
#R
#E
#A
#L


