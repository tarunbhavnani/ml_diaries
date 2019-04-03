#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:50:22 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

#data prep
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, MaxPooling1D, GlobalAveragePooling1D,SeparableConv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd




#tok.fit_on_texts(fdf.Des.values)
dat= fdf[fdf["classification"]!="Not_Tagged"]
#newdf["Des"]=[str(i) for i in newdf["Des"]]

#######
le = LabelEncoder()
dat["labels"]=le.fit_transform(dat.classification)
####

X_train, X_test, y_train, y_test = train_test_split(dat,dat["labels"], test_size=0.15, random_state=42)

#Initiate the tokenizer with all the values

tok= Tokenizer(num_words=2000, split=" ", oov_token="-OOV-")
tok.fit_on_texts(dat["Des"])


#for X_train
X_tr=tok.texts_to_sequences(X_train["Des"])

X_tr=pad_sequences(X_tr, maxlen=10)

X_tt=tok.texts_to_sequences(X_test["Des"])

X_tt=pad_sequences(X_tt, maxlen=10)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

Y=pd.get_dummies(y_train.values)

#Model!!


op_units, op_activation = len(set(dat["labels"])), "softmax"
    
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, MaxPooling1D, GlobalAveragePooling1D, SeparableConv1D    

model = Sequential()
model.add(Embedding(input_dim=2000, output_dim= 128,
                        input_length= X_tr.shape[1]))
model.add(Dropout(rate=.2))
model.add(SeparableConv1D(filters=32, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))
    
model.add(SeparableConv1D(filters=32, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))
model.add(MaxPooling1D())
    #model.add() 
model.add(SeparableConv1D(filters=64, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))
model.add(SeparableConv1D(filters=64, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))
model.add(GlobalAveragePooling1D())
#model.add(Flatten())
model.add(Dropout(rate=.2))
model.add(Dense(op_units, activation=op_activation))
model.summary()
    

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer= "adam")

history= model.fit(X_tr,Y, epochs=10, batch_size=32, validation_split=.33)
hist = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=hist['val_acc'][-1], loss=hist['val_loss'][-1]))
#Validation accuracy: 0.996681720325677, loss: 0.021983623765626834

#############Plotting#############3
# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#now lets predict and check

y_prob = model.predict(X_tt) 
y_classes = y_prob.argmax(axis=-1)

X_test["predict"]= le.inverse_transform(y_classes)


from sklearn.metrics import confusion_matrix

confusion_matrix(y_true=X_test["classification"], y_pred=X_test["predict"])



