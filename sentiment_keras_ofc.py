#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:33:41 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
import os
os.chdir('/home/tarun.bhavnani/Desktop/git_tarun/ml_diaries')
os.listdir()

import pandas as pd
dat=pd.read_csv('Sentiment.csv',encoding="latin1")
list(dat)
  
data=dat[['text','sentiment']]

import re

data["text"]= data["text"].apply(lambda x: x.lower())
data["text"]= data["text"].apply(lambda x: re.sub(r"[^a-zA-Z0-9]"," ",x))
data["text"]= data["text"].apply(lambda x: re.sub(r'\brt\b'," ",x))
data["text"]= data["text"].apply(lambda x: x.strip())

#tokenzie
from keras.preprocessing.text import Tokenizer

tok= Tokenizer(num_words=2000, split=" ", oov_token="-oov-")
tok.fit_on_texts(data["text"].values)

X= tok.texts_to_sequences(data["text"].values)


from keras.preprocessing.sequence import pad_sequences
X= pad_sequences(X)

#tok.index_word

tok.word_index["-oov-"]

y= pd.get_dummies(data["sentiment"])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=.33, random_state=1)
##data prep done, now lets begin model!!


from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D, BatchNormalization, Embedding, Bidirectional

model= Sequential()
model.add(Embedding(input_dim=2000 , output_dim=128 , input_length= X.shape[1]))

model.add(SpatialDropout1D(rate=.2))

model.add(LSTM(units=128, dropout=.1,recurrent_dropout=.1))

model.add(Dense(3, activation= "softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer="adam")

model.fit(x_train, y_train, validation_split=.2,epochs=20, batch_size=32,verbose=1 )


#evaluate

model.evaluate(x_test, y_test)
#[1.577817829559338, 0.6546526867627785]

#lets try the same with character level




#bidirectional
model = Sequential()

model.add( Embedding(input_dim=2000, output_dim = 128, input_length = X.shape[1], dropout=0.2))
model.add( Bidirectional( LSTM(units = 196, dropout_U = 0.2, dropout_W = 0.2)))
model.add( Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_split=.2,epochs=20, batch_size=32,verbose=1 )

model.evaluate(x_test, y_test)
#[1.55189290794419, 0.6408912188728703]




#lets use attention

import keras
from keras_self_attention import SeqSelfAttention


model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=2000,
                                 output_dim=300,input_length= X.shape[1],
                                 mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,return_sequences=True)))

model.add(keras.layers.LSTM(units=128,return_sequences=True))

model.add(SeqSelfAttention(attention_activation='sigmoid'))
#model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=3, activation="softmax"))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.summary()

model.fit(x_train, y_train, validation_split=.2,epochs=20, batch_size=32,verbose=1 )


#doesn't work as return sequences are true!! it works for transalation and not classification



from keras.layers import TimeDistributed
##not working lets see just a bidirectional lstm as is
model = Sequential()
model.add(Embedding(input_dim=2000 , output_dim=128 , input_length= X.shape[1]))

model.add(Bidirectional(LSTM(20, return_sequences=True)))
model.add(TimeDistributed(Dense(3, activation='softmax')))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, validation_split=.2,epochs=20, batch_size=32,verbose=1 )


#####################################################33old one#################################333

data['text'] = data['text'].apply(lambda x: x.lower())
import re
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
data['text']= [x.encode('ascii') for x in data['text']]
data['text']= [x.decode('utf-8') for x in data['text']]#to remove prefix b


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 2500, split=' ')
tokenizer.fit_on_texts(data['text'].values)
#print(tokenizer.word_index)  # To see the dicstionary
X = tokenizer.texts_to_sequences(data['text'].values)
tokenizer.word_index
max([len(i) for i in X])

X = pad_sequences(X)

embed_dim = 128
lstm_out = 300
batch_size= 32



from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, LSTM, InputLayer, Bidirectional,SpatialDropout1D

model=Sequential()

model.add(Embedding(input_dim=2500, output_dim=embed_dim,input_length=X.shape[1]))

model.add(SpatialDropout1D(.1))

model.add(LSTM(units=lstm_out, dropout=.1, recurrent_dropout=.1))

""" two lstm layers
model.add(LSTM(units=300,return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
model.add(LSTM(units=500, dropout=0.1, recurrent_dropout=0.1))
"""

model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', metrics=['accuracy'],loss='categorical_crossentropy')

#get one hot encodes for the output

Y=pd.get_dummies(data['sentiment']).values

from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)


model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=1, verbose=1)

print(model.summary())