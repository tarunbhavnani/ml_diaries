#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:33:41 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
import os
os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/keras')
os.listdir()

dat=pd.read_csv('Sentiment.csv',encoding="latin1")
  
data=dat[['text','sentiment']]


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