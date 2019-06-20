#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:57:04 2019

@author: tarun.bhavnani

"""

#get mnist data
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


#plt.imshow(x_train[7777])
#clear

#we have to comver it to grey scale and also 3d
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





########################33
#autoencoders
#autoencoder rebuilds the same image after compression thus learning features.

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


input_image= Input(shape=(28,28,1))
x= Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same")(input_image)
x= MaxPooling2D(padding="same")(x)
x= Conv2D(filters=8, kernel_size=(3,3), activation="relu", padding="same")(x)
x= MaxPooling2D(padding="same")(x)
x= Conv2D(filters=8, kernel_size=(3,3), activation="relu", padding="same")(x)
encoded= MaxPooling2D(padding="same", name="encoder")(x)

x= Conv2D(filters=8, kernel_size=(3,3), activation="relu",padding="same")(encoded)
x= UpSampling2D((2,2))(x)
x= Conv2D(filters=8, kernel_size=(3,3), activation="relu",padding="same")(x)
x= UpSampling2D((2,2))(x)
x= Conv2D(filters=16, kernel_size=(3,3), activation="relu")(x)
x= UpSampling2D((2,2))(x)
decoded=Conv2D(filters=1, kernel_size=(3,3), activation="sigmoid", padding="same")(x)


autoencoder= Model(input_image, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(x=x_train,y=x_train, batch_size=32, epochs=10, validation_split=.3)


##
##




input_img = Input(shape=(28,28,1))
x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
autoencoder.fit(x=x_train,y=x_train, batch_size=32, epochs=10, validation_split=.3)

#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
#encoder.fit(x=x_train,y=x_train, batch_size=32, epochs=10, validation_split=.3)
x_train[1]
plt.imshow(x_train[1].reshape(28,28))

asd=autoencoder.predict(x_train[1].reshape(1,28,28,1))
plt.imshow(asd.reshape(28,28) )

#at ten epochs it shows shit


####view

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
























