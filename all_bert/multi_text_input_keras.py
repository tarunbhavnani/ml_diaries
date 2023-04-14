# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:07:21 2021

@author: ELECTROBOT
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split


inp1 = Input(shape=(100, ))
x = Embedding(input_dim=len(tok.word_index)+1, output_dim= 256)(inp1)
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
    
x1= concatenate(layers, axis=-1)
    

inp2 = Input(shape=(100, ))
x = Embedding(input_dim=len(tok.word_index)+1, output_dim= 256)(inp2)
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
    
x2= concatenate(layers, axis=-1)
    
    
    
    
#inputs are 5
final=concatenate([x1]+[x2], axis=-1)
        
    
x = Dropout(0.1)(final)
x = Dense(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5))(x)
x = Dropout(0.1)(x)
out = Dense(2, activation='softmax')(x)
model = Model([inp1, inp2], outputs=out)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


