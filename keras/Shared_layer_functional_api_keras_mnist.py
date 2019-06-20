#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:57:04 2019

@author: tarun.bhavnani
https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras

for _ in range(np_epochs):
    model.fit(X_train, Y_train, batch_size=1,nb_epoch=1, verbose=1)
    print_image()
    
"""

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


#y_train[7777]

import matplotlib.pyplot as plt

plt.imshow(x_train[7777])

x_train.shape
#clear

# we have to convert it to grey scale and also 3d
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


#shared layers using keras functional api!

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Input, concatenate
from keras.layers import GlobalAveragePooling2D

input1= Input(shape=(28,28,1))
input2= Input(shape=(28,28,1))

conv2d3= Conv2D(filters=128, kernel_size=(3,3), padding="same")
conv2d5= Conv2D(filters=128, kernel_size=(5,5), padding="same")

maxpool= MaxPooling2D()
globalPool= GlobalAveragePooling2D()
flat= Flatten()

encode1=conv2d3(input1)
encode2=conv2d5(input2)

#can we use this for multiple images of same dresses.


merged_vec= concatenate([encode1, encode2], axis=-1)

merged_vec= maxpool(merged_vec)

flat_vec= flat(merged_vec) #first epoch just after start it reched 6-70 pc casually and ends 1st epoch >90 pc acc
#flat_vec= globalPool(merged_vec) #2nd epoch start at .45 acc

out= Dense(128, activation="relu")(flat_vec)

out2=Dropout(.2)(out)

out3= Dense(len(set(y_train)), activation="softmax")(out2)


#out3= Dense(len(set(y_train)), activation="softmax")(merged_vec)
#model = Model(inputs=[encode1, encode2], outputs=predictions)

mdl= Model(inputs=[input1, input2],outputs= out3)
#mdl= Model(input1, out3)

#mdl.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#not working this, see why

mdl.compile(loss='sparse_categorical_crossentropy',  optimizer="adam",metrics=["acc"] )
mdl.summary()
hist1=mdl.fit([x_train, x_train], y_train, epochs=10, batch_size=64, validation_split=.3)

#mdl.compile(loss='categorical_crossentropy',  optimizer="adam",metrics=["acc"] )
#mdl.summary()
#hist1=mdl.fit([x_train, x_train], pd.get_dummies(y_train), epochs=10, batch_size=64, validation_split=.3)




#sequential

#build model
#Importing the required Keras modules containing model and layers
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Input
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(set(y_train)),activation="softmax"))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',  optimizer="adam",metrics=["acc"] )

hist2=model.fit(x=x_train,y=y_train, epochs=10,batch_size=64, validation_split=.3)

#model.compile(loss='categorical_crossentropy',  optimizer="adam",metrics=["acc"] )

#hist2=model.fit(x=x_train,y=pd.get_dummies(y_train), epochs=10,batch_size=64, validation_split=.3)

