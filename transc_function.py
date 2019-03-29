
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:23:08 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
import pandas as pd
import os
import re



def clean_transc(dat):
  
  dat["Des"] = [str(i) for i in dat["Description"]]
  dat["Des"] = dat["Des"].apply(lambda x: x.lower())
  dat["Des"]=[re.sub("[i|1]/w"," inwards ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("[o|0]/w"," outwards ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("b/f"," brought_fwd ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("neft"," neft ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("[i|1]mp[s|5]"," imps ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub(r'r[i|t|1][g|8][s|5]'," rtgs ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("ecs"," ecs ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("cash"," cash ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("nach"," nach ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("ebank"," ebank ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub(r"c[o|0][1|l][1|l]","coll",i) for i in dat["Des"]]# for int.co11
  dat["Des"]=[re.sub("vvdl","wdl",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("nfs"," nfs ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub(r"[1|l][o|0]an","loan",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("[\n]|[\r]"," ",i) for i in dat["Des"]]


  dat["Des"]=[re.sub(r"\(|\)|\[|\]"," ",i) for i in dat["Des"]]#brackets

  dat["Des"]=[re.sub("-|:|\.|/"," ",i) for i in dat["Des"]]


  dat["Des"]=[re.sub("a c ","ac ", i) for i in dat["Des"]]




  dat["Des_cl"]= [re.sub(" ","",i) for i in dat["Des"]]

  #########################################################################
  "Classification"



  dat["classification"]="Not_Tagged"


  dat["classification"]=["dd" if len(re.findall(r"\bdd\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["ib" if len(re.findall(r"\bib\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["ft" if len(re.findall(r"\bft\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]


  dat["classification"]=["brought_fwd" if len(re.findall("brought_fwd",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]




  dat["classification"]=["transfer" if len(re.findall(r"\bdr\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]


  dat["classification"]=["transfer" if len(re.findall("tpt",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]


  dat["classification"]=["transfer" if len(re.findall("transfer",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["neft" if len(re.findall("neft",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["rtgs" if len(re.findall("rtgs",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]


  dat["classification"]=["imps" if len(re.findall(r"[i|1]mps",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("cash",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("at[m|w]",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("self",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall(r"\bpos\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("debitcard",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cheque" if len(re.findall(r"che?q",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cheque" if len(re.findall("cl[g|q]",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cheque" if len(re.findall(r"c[l|1]earing",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  
  
  dat["classification"]=["INF" if len(re.findall(r"\b[i|1]nf\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["MMT" if len(re.findall(r"\bmmt\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  
  
  dat["classification"]=["ecs" if len(re.findall("ecs",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["ecs" if len(re.findall("loan",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["emi" if len(re.findall(r"\bemi\b",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["nach" if len(re.findall("nach",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["nach" if len(re.findall(r"\bach\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["i/w" if len(re.findall("inward",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  dat["classification"]=["o/w" if len(re.findall("outward",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  dat["classification"]=["int_coll" if len(re.findall("int coll",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  #charges!
  dat["cl_cl"]="Not_Tagged"
  dat["cl_cl"]=["charges" if len(re.findall(r"charge?",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall("chrg",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall("chgs?",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall("commission",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall(r"\bfee\b",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  #return
  #dat["cl_ret"]="Not_Tagged"
  #dat["cl_cl"]=["return" if len(re.findall(r"\bretu?r?n?|return",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  dat["cl_cl"]=["return" if len(re.findall(r"\bretu?r?n?\b|return",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]

  dat["gst"]="Not_Tagged"
  dat["gst"]=["gst" if len(re.findall(r"[s|c]gst|\bgst",x))>0 else y for x,y in zip(dat["Des"], dat["gst"])]



  dat["Des"]=[" ".join([j for j in i.split() if not any(c.isdigit() for c in j)]) for i in dat["Des"]]
  #if any alpha numerics are still left

  dat["Des"]= [re.sub("[\W_]+"," ",i) for i in dat["Des"]]

  return(dat)




fdf["classification"].value_counts()
#dat["tt"].value_counts()
#dat.to_csv("finaldf8.csv")


#fdf["Des"][350838]

#once classification via regex is done, we will chunk out the tagged.
#for the rest of the Not_Tagged, we will create a model to predict
# this will work on clean description the "Des" coloumn
#we will sue sepcnn

os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/ocr_trans/all mapped')
fdf= pd.read_csv("finaldf.csv")

list(fdf)
fdf= fdf.drop(["classification", "Des"], axis=1)

%time fdf= clean_transc(fdf)

#timing: 350935 recorde 21.8 secsonds

fdf["Des"]

dat= fdf[fdf["classification"]=="Not_Tagged"]
dat= dat[dat["cl_cl"]=="Not_Tagged"]
dat= dat[dat["gst"]=="Not_Tagged"]
dat.shape
#or
#dat= fdf[fdf["classification"]=="Not_Tagged"][fdf["cl_cl"]=="Not_Tagged"][[fdf["gst"]=="Not_Tagged"]]


dat["Description"][350809]


dropout_rate=.2
input_shape=(None,256)
units=128
final_units=5
from keras.models import Sequential
from keras.layers import Dropout

model= Sequential()
model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

model.add(Dense(units, activation="relu"))
model.add(Dropout(rate=dropout_rate))


model.add(Dense(final_units, activation="softmax"))
model.summary()



###################################################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SeparableConv1D, MaxPooling1D,GlobalAveragePooling1D

num_features=200
embedding_dim=128
input_shape=(None, 256)
model= Sequential()
filters=128
kernel_size=3

model.add(Embedding(input_dim=num_features,
                    output_dim=embedding_dim,
                    input_length=input_shape[0]))

model.add(Dropout(rate= dropout_rate))
model.add(SeparableConv1D(filters=filters,
                          kernel_size=kernel_size,
                          activation="relu",
                          bias_initializer="random_uniform",
                          depthwise_initializer="random_uniform",
                          padding="same"))

model.add(SeparableConv1D(filters=filters,
                          kernel_size=kernel_size,
                          activation="relu",
                          bias_initializer="random_uniform",
                          depthwise_initializer="random_uniform",
                          padding="same"))

model.add(MaxPooling1D(pool_size=pool_size))

model.add(SeparableConv1D(filters=filters*2,
                          kernel_size=kernel_size,
                          activation="relu",
                          bias_initializer="random_uniform",
                          depthwise_initializer="random_uniform",
                          padding="same"))

model.add(SeparableConv1D(filters=filters*2,
                          kernel_size=kernel_size,
                          activation="relu",
                          bias_initializer="random_uniform",
                          depthwise_initializer="random_uniform",
                          padding="same"))
model.add(GlobalAveragePooling1D())

model.add(Dropout(rate=dropout_rate))

model.add(Dense(3, activation="softmax"))
model.summary()











