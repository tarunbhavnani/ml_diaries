#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:50:22 2019

@Author: tarun.bhavnani@dev.smecorner.com
"""

#data prep
import pandas as pd
import os
import re
os.chdir("/home/tarun.bhavnani@dev.smecorner.com/Desktop/ocr_trans")

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, MaxPooling1D, GlobalAveragePooling1D,SeparableConv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from transc_function import clean_transc
from transc_function import pattern
from keras.callbacks import EarlyStopping

fdf=pd.read_csv("finaldf8.csv")
#from transc_function import clean_transc 
fdf=clean_transc(fdf)


#tok.fit_on_texts(fdf.Des.values)
dat= fdf[fdf["classification"]!="Not_Tagged"]
#dat["labels"]

#newdf["Des"]=[str(i) for i in newdf["Des"]]

#######
le = LabelEncoder()
dat["labels"]=le.fit_transform(dat.classification)
#jk= dat.pop("labels")
####
Y=pd.get_dummies(dat["labels"])

X_train, X_test, y_train, y_test = train_test_split(dat,Y, test_size=0.15, random_state=43)

#Initiate the tokenizer with all the values

tok= Tokenizer(num_words=2000, split=" ", oov_token="-OOV-")
tok.fit_on_texts(dat["Des"])


#for X_train
X_tr=tok.texts_to_sequences(X_train["Des"])

X_tr=pad_sequences(X_tr, maxlen=10)

X_tt=tok.texts_to_sequences(X_test["Des"])

X_tt=pad_sequences(X_tt, maxlen=10)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

Y_tr=y_train
Y_tt=y_test

#Model!!


op_units, op_activation = len(set(dat["labels"])), "softmax"
    
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, MaxPooling1D, GlobalAveragePooling1D, SeparableConv1D, Flatten

model = Sequential()
model.add(Embedding(input_dim=2000, output_dim= 128,input_length= X_tr.shape[1]))
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
#GAP does this: (None, 5, 64)    ---->(None, 64)    
#model.add(Flatten())
#flatten does this: (None, 5, 64) --->(None, 320) 
model.add(Dropout(rate=.2))
model.add(Dense(op_units, activation=op_activation))
model.summary()
    

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer= "adam")

from time import time

t0 = time()
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
history= model.fit(X_tr,Y_tr, epochs=10, batch_size=32, validation_split=.33,callbacks=[early_stop])
print("done in %0.3fs" % (time() - t0))

hist = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=hist['val_acc'][-1], loss=hist['val_loss'][-1]))


score = model.evaluate(X_tt, Y_tt, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

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

X_test.to_csv("testdf.csv")

from sklearn.metrics import classification_report
print(classification_report(X_test["final_tag3"],X_test["predict"]))
print(confusion_matrix(X_test["classification"],X_test["predict"]))


X_test["chk"]=[1 if i==j else 0 for i,j in zip(X_test["final_tag3"], X_test["predict"])]



########################################

#Save Model

# serialize model to JSON
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer= "adam")
score = loaded_model.evaluate(X_tt, Y_tt, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

