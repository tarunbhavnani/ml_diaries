#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:43:22 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

#loading and classifing transaction analysis
import os
os.chdir("/home/tarun.bhavnani/Desktop/ocr_trans/Final_Models/new")

from transc_function import clean_transc
from keras.models import model_from_json
import pickle
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from time import time
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt

#del dat1
#######################
###Input data file!!###
#######################
dat1= pd.read_excel("405201907_Final.xlsx", sheet_name="All_Transaction")



##clean and tag regex
#dat1=dat1[["Description","Current","Change "]]

dat1= clean_transc(dat1)
#dat1.Des[0:10]
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

dat1["classification"]= [chngs[i] if i in chngs else i for i in dat1["classification"]]

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
tok.word_index["-OOV-"]=tok.word_index["i"]

X=tok.texts_to_sequences(dat1["Des"])

X=pad_sequences(X, maxlen=10)


encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)




###########now lets predict and check


y_prob = loaded_model.predict(X) 
y_classes = y_prob.argmax(axis=-1)

dat1["predict"]= encoder.inverse_transform(y_classes)

dat1["classification"]=[y if x=="Not_Tagged" else x for x,y in zip(dat1["classification"], dat1["predict"])]


#chngs={"number":"others","nach":"nach/emi","emi":"nach/emi","charge":"charges","o/w returnreturn":"o/wreturn","Tax":"tax","Transfer":"transfer","IMPS":"imps","sudhircharges": "charges","Fund Transfer": "transfer", "ecs":"nach/emi", "mmt": "cash","sudhirreturn": "return" }

#dat1["classification"]= [chngs[i] if i in chngs else i for i in dat1["classification"]]

dat1["disbursement"]=[1 if y>z and x=="nach/emi"else 0 for x,y,z in zip(dat1["classification"], dat1["Credit"], dat1["Debit"]) ]
dat1["classification"]=["disbursement" if y==1 else x for x,y in zip(dat1["classification"], dat1["disbursement"])]

dat1["interest"]=[1 if y>z and x=="int_coll"else 0 for x,y,z in zip(dat1["classification"], dat1["Credit"], dat1["Debit"]) ]
dat1["classification"]=["int_coll_credit" if y==1 else x for x,y in zip(dat1["classification"], dat1["interest"])]


#int_coll_credit is when ibt coll is in credit
#disbursement is when nach/emi is in credit


dat1=dat1.drop(labels=["cl_cl","Des_cl","predict","interest", "disbursement"], axis=1)



####save final data file#############
dat1.to_csv("Final_data.csv")



#list(dat1)


"""
#left:

int_coll--> further segregate if debit or credit
nach/emi--> further segregate, if debit then emi, if credit then disbursal


"""


