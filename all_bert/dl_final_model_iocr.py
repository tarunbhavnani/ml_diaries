#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:18:02 2020

@author: tarun.bhavnani
"""

import os
import re
import pandas as pd
import numpy as np
import glob
os.chdir('/home/tarun.bhavnani/Desktop/ocr_trans_may5/phase2')
from trans_class import *
path='/home/tarun.bhavnani/Desktop/ocr_trans_may5/final_codes/deploy_codes'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gc
file_list = glob.glob("/home/tarun.bhavnani/Desktop/ocr_trans_may5/phase2/final_model_Description/clean_data/*")
#os.chdir('/home/tarun.bhavnani/Desktop/ocr_trans_may5/phase2/final_model_Description/clean_data')
#os.chdir('/home/tarun.bhavnani/Desktop/ocr_trans_may5/phase2/final_model_Description')

#final_X=[]
#final_label=[]
label={}
label_rev={}

#Debit=[]
#Credit=[]

counter=0
#for i in file_list[22:]:
#for i in file_list[12:22]:
for i in file_list:
    print(i)
    print(counter)
    dat= pd.read_csv(i)
    #intel=iocr(dat=dat.copy(), path=path)
    #clean_dat=intel.clean()
    #clean_dat.to_csv(i.split('/')[-1], index=False)
    clean_dat=dat[dat.classification!="Not_Tagged"]
    tags= ['card swipe','cash','charges','credit card payment','emi','emi bounce','i/w','i/w return',
           'interest','investment','loan disbursement','o/w','o/w return','refund','rent','return',
           'reversal','salary','statutory dues','transfer','upi','utility']
    clean_dat= clean_dat[clean_dat.classification.isin(tags)]
    if counter==0:
        tok= Tokenizer(char_level=True)
        tok.fit_on_texts(clean_dat.Des.values)
        
        for i,j in enumerate(sorted(set(clean_dat.classification))):
            label[j]=i
            label_rev[j]=i
        print(len(label))
            
    X= tok.texts_to_sequences(clean_dat.Description.values)
    X= pad_sequences(X, maxlen=100)
    
    labels= [label[i] for i in clean_dat.classification]
    
    bins = [-10000000, 0, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    bin_labels = [1,2,3,4,5,6,7]
    Debit = pd.cut(clean_dat['Debit'], bins=bins, labels=bin_labels)
    Credit = pd.cut(clean_dat['Credit'], bins=bins, labels=bin_labels)
    
    
    
    if counter==0:
        final_X= X
        final_label=labels
        final_debit= list(Debit.values)
        final_credit= list(Credit.values)
    else:
        final_X=np.vstack([final_X, X])
        final_label= final_label+labels
        final_debit= final_debit+list(Debit.values)
        final_credit= final_credit+list(Credit.values)
    
    del dat, clean_dat
    gc.collect()
    counter+=1
    
    

#import pickle
with open('final_tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('final_label.pickle', 'wb') as handle:
    pickle.dump(final_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('final_X.pickle', 'wb') as handle:
    pickle.dump(final_X, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('final_debit.pickle', 'wb') as handle:
    pickle.dump(final_debit, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('final_credit.pickle', 'wb') as handle:
    pickle.dump(final_credit, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('label.pickle', 'wb') as handle:
    pickle.dump(label, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
#import pickle    
with open( 'final_X.pickle', "rb") as f:
    final_X=pickle.load( f)
#with open("final_label.pkl", "wb") as f:
#    pickle.dump(final_label, f)
#

#with open(PIK, "rb") as f:
#    print pickle.load(f)


Input_final= np.asarray(final_X)
#Input_other=np.column_stack((final_debit, final_credit))


# =============================================================================
# #this solves the loss nan issue.. crazy
# =============================================================================
final_debit=[1 if i>0 else 0 for i in final_debit]
final_credit=[1 if i>0 else 0 for i in final_credit]
Input_other=np.column_stack((final_debit, final_credit))


from keras.layers import Embedding, Input, Dropout, Dense, concatenate, Flatten, GlobalMaxPool1D,Conv1D
from keras.models import Model
from keras import regularizers

inp = Input(shape=(100, ))
x = Embedding(input_dim=len(tok.word_index)+1, output_dim= 256)(inp)
x = Dropout(0.25)(x)

layers=[]
x1 = Conv1D(16, kernel_size = 3, activation = 'relu')(x)
x1 = Conv1D(32, kernel_size = 3, activation = 'relu')(x1)
x1=Dropout(.5)(x1)
x1= GlobalMaxPool1D()(x1)
layers+=[x1]

x2 = Conv1D(16, kernel_size = 4, activation = 'relu')(x)
x2 = Conv1D(32, kernel_size = 4, activation = 'relu')(x2)
x2=Dropout(.5)(x2)
x2= GlobalMaxPool1D()(x2)
layers+=[x2]

x3 = Conv1D(16, kernel_size = 5, activation = 'relu')(x)
x3 = Conv1D(32, kernel_size = 5, activation = 'relu')(x3)
x3=Dropout(.5)(x3)
x3= GlobalMaxPool1D()(x3)
layers+=[x3]

x= concatenate(layers, axis=-1)

#x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5))(x)
x = Dropout(0.1)(x)

inp_other= Input(shape=(2,))
#inp_other= Input(shape=(2,))
#y=Dense(4, activation='relu')(inp_other)
y=Dense(2, activation='relu')(inp_other)


x= concatenate([x]+[y], axis=-1)

x = Dropout(0.1)(x)
x = Dense(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5))(x)
x = Dropout(0.1)(x)
out = Dense(len(tags), activation='softmax')(x)
model = Model([inp,inp_other], outputs=out)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#from sklearn.model_selection import train_test_split

#X_train, X_test, Y_train, Y_test= train_test_split(final_X, final_label, stratify=final_label, test_size=.3)
#how to put debit and credit in this?
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(final_label),
                                                 final_label)



from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
history= model.fit([Input_final,Input_other],pd.get_dummies(final_label),
                   epochs=3, batch_size=128, validation_split=.30,callbacks=[early_stop],
                   class_weight=class_weights, shuffle=True)


#remove validation split and train on all data for 2 epochs
#history1= model.fit([Input_final,Input_other],pd.get_dummies(final_label),
                   epochs=2, batch_size=128,callbacks=[early_stop],
                   class_weight=class_weights, shuffle=True)


# =============================================================================
# #next try: out of the 27 labels just choose the labels which needs to be used in the model.
# #for eg: neft, rtgs, imps, reversal, and a few more can be removed, as they will be tagged in regx for sure.
# #unless you want to skip the regex model for good.
# 
# =============================================================================



#save model, save history
from keras.models import model_from_json
model_json = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_final.h5")
print("Saved model to disk")


import matplotlib.pyplot as plt
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_acc'])

#3 epochs is the sweet spot

#lets test on NOt_Tagged
dat_check=pd.DataFrame()
for i in file_list[-10:]:
    print(i)
    dat= pd.read_csv(i)
    dat= dat[dat.classification=="Not_Tagged"]
    dat_check=dat_check.append(dat)


X_test= tok.texts_to_sequences(dat_check.Description.values)
X_test= pad_sequences(X_test, maxlen=100)
X_test=np.asarray(X_test)

Debit_test= [1 if i>0 else 0 for i in dat_check.Debit]
Credit_test= [1 if i>0 else 0 for i in dat_check.Credit]
X_test_other=np.column_stack([Debit_test, Credit_test])


pred=model.predict([X_test, X_test_other])

#dat_check['pred']= np.argmax(pred, axis=1)
#dat_check=[label[i] for i in dat_check['pred']]
#label_rev={j:i for i,j in label.items()}
dat_check['predicted']=[label_rev[np.argmax(p)]  for p in pred]
dat_check['predicted_pc']=[ max(p)  for p in pred]

dat_check['decile']=pd.qcut(dat_check['predicted_pc'].rank(method='first', ascending=False), 10, labels=False)

dat_check.groupby("decile").predicted_pc.min()
dat_c= dat_check[dat_check.decile<=2]


# =============================================================================
# #good till 2nd decile at the max, or 85+ pc
# =============================================================================


#how to proceed
#1st approach
#regex model is run
#all with not tagged are passed through this model , all with prob of >85~90 pc are tagged(check)

# last we get new predictions on not tagged from the emi model from char_model5

#combine

#2nd app
#run this model
#run char_model5
#results


#lets check on NOt_Tagged













