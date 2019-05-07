#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:43:22 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

#loading and classifing transaction analysis
import pandas as pd
import os
#from transc_function import clean_transc
from keras.models import model_from_json
import pickle
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
from time import time
import re

os.chdir("/home/tarun.bhavnani@dev.smecorner.com/Desktop/ocr_trans/Final_Models/Trans_Final_Model")


dat= pd.read_excel("data/4052019003_Final.xlsx", sheet_name="All_Transaction")

#call the function clean_transc from transc_function.py
dat= clean_transc(dat)
#dat.classification.value_counts()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
loaded_model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer= "adam")
###########################################################

#load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)
#removing oov and changing it to the location of something irrelevant as "i" also because i is at 71 well within 2000
#tok.word_index["-OOV-"]=tok.word_index["i"]
#tok.word_index["-OOV-"]

X=tok.texts_to_sequences(dat["Des"])
X=pad_sequences(X, maxlen=10)

#load the labelencoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy')


#now lets predict and check

y_prob = loaded_model.predict(X) 
y_classes = y_prob.argmax(axis=-1)

dat["predict"]= encoder.inverse_transform(y_classes)

dat["classification"]=[y if x=="Not_Tagged" else x for x,y in zip(dat["classification"], dat["predict"])]
dat.classification.value_counts()

#chngs={"number":"others","nach":"nach/emi","emi":"nach/emi","charge":"charges","o/w returnreturn":"o/wreturn","Tax":"tax","Transfer":"transfer","IMPS":"imps","sudhircharges": "charges","Fund Transfer": "transfer", "ecs":"nach/emi", "mmt": "cash","sudhirreturn": "return" }
#dat["classification1"]= [chngs[i] if i in chngs else i for i in dat["classification"]]

dat=dat.drop(["predict"], axis=1)

dat.to_csv("data/4052019003_Final.csv")








