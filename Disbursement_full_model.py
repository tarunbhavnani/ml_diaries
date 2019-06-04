#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:23:08 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
import pandas as pd
import os
os.chdir('/home/tarun.bhavnani/Desktop/disburse/bank_data')

fdf= pd.read_excel('output0_50200012559334_Final.xlsx', sheet_name="All_Transaction")
#fna= pd.read_excel('output0_50200012559334_Final.xlsx', sheet_name="All_Transaction")

#nm= list(fdf)
names=['Date_new','Description','Credit','Debit','Balance']

fdf=fdf[names]

fdf["counter"]=1


c=2
not_added=[]
for i in os.listdir():
 try:   
  print(c)
  if c<1001:
    if "xlsx" in i:
      df= pd.read_excel(i, sheet_name="All_Transaction")
      df=df[names]
      df["counter"]=c
      #if list(df)==names:
      fdf=fdf.append(df)
      c+=1
      #else: 
      #  print("headers name not match")
      #  not_added.append(i)
    #else:
    #  print("not xslx")
 except:
  print("some error")   
    

fdf=fdf.reset_index(drop=True)
fdf.to_csv("top_1000_recs_bank_data.csv")

fdf_cr= fdf[fdf["Credit"]>20000]

from time import time
import re

fdf1= clean_transc(fdf)



#reading data created by sudhir

dat= pd.read_excel("/home/tarun.bhavnani/Desktop/disburse/sudhir.xlsx")
dat_db= dat[dat["disbursement"]==1]

#upsampling
dat=dat.append(dat_db)
dat=dat.append(dat_db)
dat=dat.append(dat_db)
dat=dat.append(dat_db)
dat=dat.append(dat_db)
dat=dat.append(dat_db)


dat1= clean_transc(dat)

X= dat1["Des"]

from keras.preprocessing.text import Tokenizer

tok= Tokenizer(num_words=2000, lower=True,split=" ", oov_token="-OOV-")
tok.fit_on_texts(X)

tok.word_index["-OOV-"] #just check if number is greater then 2000, get it under 2000

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(dat1["Des"], dat1["disbursement"], test_size=.2, random_state=2)

lent=[len(i) for i in dat1["Des"]]
import matplotlib.pyplot as plt
plt.plot(lent)

X_tr= tok.texts_to_sequences(X_train)
from keras.preprocessing.sequence import pad_sequences
X_tr= pad_sequences(X_tr, maxlen=75)



#model
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
model.add(Dense(1, activation="sigmoid"))
model.summary()
    

model.compile(loss="binary_crossentropy", metrics=["acc"], optimizer= "adam")

from keras.callbacks import EarlyStopping
#t0 = time()
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
history= model.fit(X_tr,Y_train, epochs=10, batch_size=64, validation_split=.15,callbacks=[early_stop])



#Save Model

# serialize model to JSON
from keras.models import model_from_json
model_json = model.to_json()
with open("model_disb.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_disb.h5")
print("Saved model to disk")


#evaluate
X_tt= tok.texts_to_sequences(X_test)
#from keras.preprocessing.sequence import pad_sequences
X_tt= pad_sequences(X_tt, maxlen=75)



for threshold in [.25,.35,.45,.50,.6,.7,.8,.9]:
    #print(threshold)
    evalu= model.predict(X_tt)
    evalu=[1 if i>threshold else 0 for i in evalu ]


    #manual confusion matrix creation
    dfg= pd.DataFrame(dict(real=Y_test, predict=evalu))
    dfg=dfg.reset_index(drop=True)


    #["TP" if i==j==1 else ("TN" if i==j=0 else ("FP" if i==1 and j==0 else "FN")) for i,j in zip(dfg["real"], dfg["predict"])]
    dfg["TP"]=[1 if i==1 and j==1 else 0 for i,j in zip(dfg["real"], dfg["predict"])]
    dfg["FP"]=[1 if i==1 and j==0 else 0 for i,j in zip(dfg["real"], dfg["predict"])]
    dfg["FN"]=[1 if i==0 and j==1 else 0 for i,j in zip(dfg["real"], dfg["predict"])]
    dfg["TN"]=[1 if i==0 and j==0 else 0 for i,j in zip(dfg["real"], dfg["predict"])]


    Precision= sum(dfg["TP"])/(sum(dfg["TP"])+sum(dfg["FP"]))
    Recall= sum(dfg["TP"])/(sum(dfg["TP"])+sum(dfg["FN"]))
    f1= 2* (Precision*Recall)/(Precision+Recall)

    print(threshold, f1)


#working best at .8
    

##################################
    ##############################

#test data

df= pd.read_csv("top_1000_recs_bank_data.csv")
list(df)
df= clean_transc(df)
X_p= tok.texts_to_sequences(df["Des"])
X_p= pad_sequences(X_p, maxlen=75)

df["disbursement"]= model.predict(X_p)
df["disbursement"]= [1 if i>.8 else 0 for i in df["disbursement"] ]
df["disbursement"].value_counts()

df["disbursement"]= [0 if x==0 else y for x,y in zip(df["Credit"], df["disbursement"]) ]
df["disbursement"].value_counts()

df.to_csv("final_disbursement_data_1000_bank.csv")
dfr= df[df["disbursement"]==1]
dfr.to_csv("disburse_only_data.csv")
