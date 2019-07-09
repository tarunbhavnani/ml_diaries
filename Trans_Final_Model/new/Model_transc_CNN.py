#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:04:57 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

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
import numpy as np
import pickle
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
from time import time

#os.chdir("/home/tarun.bhavnani@dev.smecorner.com/Desktop/ocr_trans/Final_Models/new")
#first recreating the final data for model training!!
dat= pd.read_csv("/home/tarun.bhavnani/Desktop/ocr_trans/Final_Models/final_clean_data_for_training_model.csv")

list(dat)
dat= clean_transc(dat)
dat["classification"].value_counts()
#dat["classification"]=[y if y!="Not_Tagged" else x for x,y in zip(dat["classification"], dat["cl_cl"])]
dat["classification"]=[y if x=="Not_Tagged" else x for x,y in zip(dat["classification"], dat["Merged_tagging"])]

chngs={"number":"others",
       "INSUFFICIENT FUNDS":"return",
       "credit card": "creditcard",
       "nach":"nach/emi",
       "emi":"nach/emi",
       "charge":"charges",
       "o/w returnreturn":"o/wreturn",
       "Tax":"tax",
       "Transfer":"transfer",
       "IMPS":"imps",
       "sudhircharges": "charges",
       "Fund Transfer": "transfer",
       "ecs":"nach/emi",
       "third_party":"nach/emi",
       "ib":"transfer",
       "mmt": "cash",
       "sudhirreturn": "return" }

dat["classification"]= [chngs[i] if i in chngs else i for i in dat["classification"]]

#merge classes
#chngs={"number":"others","nach":"nach/emi","emi":"nach/emi","charge":"charges","o/w returnreturn":"o/wreturn","Tax":"tax","Transfer":"transfer","IMPS":"imps","sudhircharges": "charges","Fund Transfer": "transfer", "ecs":"nach/emi", "mmt": "cash","sudhirreturn": "return" }

#dat["classification"]= [chngs[i] if i in chngs else i for i in dat["classification"]]
len(set(dat["classification"]))
#23



#######
le = LabelEncoder()
dat["labels"]=le.fit_transform(dat.classification)

#save encoder model
np.save('classes.npy', le.classes_)


####
Y=pd.get_dummies(dat["labels"])

#X_train, X_test, y_train, y_test = train_test_split(dat,Y, test_size=.33, random_state=43)
X_train, X_test, y_train, y_test = train_test_split(dat,Y, test_size=0, random_state=43)#for final training

#Initiate the tokenizer with all the values

tok= Tokenizer(num_words=2000, split=" ", oov_token="-OOV-")
tok.fit_on_texts(dat["Des"])

tok.word_index["-OOV-"]
tok.word_index["i"]#71
tok.word_index["-OOV-"]=71


#save tokenizer

import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
model.add(Dropout(rate=.2))
model.add(Dense(op_units, activation=op_activation))
model.summary()
    

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer= "adam")


t0 = time()
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
history= model.fit(X_tr,Y_tr, epochs=10, batch_size=32, validation_split=.05,callbacks=[early_stop])

print("done in %0.3fs" % (time() - t0))

hist = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=hist['val_acc'][-1], loss=hist['val_loss'][-1]))





#Save Model

# serialize model to JSON
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()