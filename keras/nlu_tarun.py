#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:16:27 2019

@author: tarun.bhavnani
"""


import pandas as pd
import os
os.chdir('/home/tarun.bhavnani/Desktop/final_bot/final_bot7/data')
os.listdir()

dat= pd.read_excel("nlu.xlsx")

list(dat)
dat=dat[["- Hi","greet"]]

dat.columns=["text", "intent"]


import unicodedata
import re
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w=str(w)
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^0-9a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()
    return w

dat["text_clean"]= [preprocess_sentence(i) for i in dat["text"]]


dat1= dat[dat["intent"]!="inform"]

from keras.preprocessing.text import Tokenizer

tok= Tokenizer(num_words=2000, split=" ", oov_token="-OOV-", char_level=True)
tok.fit_on_texts(dat1["text_clean"])

X=tok.texts_to_sequences(dat1["text_clean"])
#[len(i) for i in X]
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

X= pad_sequences(X, maxlen=100)

le=LabelEncoder()
dat1["labels"]=le.fit_transform(dat1["intent"])
Y= pd.get_dummies(dat1["labels"])


from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, LSTM

#input_shape= X.shape[1]

model= Sequential()
#model.add(Input(shape= input_shape))
model.add(Embedding(2000, 128, input_length=X.shape[1]))
model.add(LSTM(128))
model.add(Dense(32, activation="relu"))
model.add(Dense(len(set(dat1["intent"])), activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer="adam")

history=model.fit(X,Y, validation_split=.3, batch_size=12, epochs=125)


#predict

pdt=model.predict(X)

dat1["predicted"]=le.inverse_transform(pdt.argmax(axis=-1))

txt="what is my name?"
txt="what is ur name?"

#import numpy as np
txt=[tok.word_index[i] for i in txt]
txt=np.asarray(txt)
txt.shape
txt= txt.reshape(1,txt.shape[0])
#txt=np.zeros((1,100))
#txt=tok.texts_to_sequences(txt)
txt=pad_sequences(txt, maxlen=100)

y_p=model.predict(txt)

predicted_intent = le.inverse_transform(y_p.argmax(axis=-1))[0]




###use attention here!!




class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
 
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
 
        return context_vector, attention_weights





max_len=100
#embed layer

sequence_input = Input(shape=(max_len,), dtype='int32')

embedded_sequences = keras.layers.Embedding(2000, 128, input_length=max_len)(sequence_input)


#bidirectional rnn

import os
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                     (128,
                                      dropout=0.3,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_activation='relu',
                                      recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)
 
lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional \
    (tf.keras.layers.LSTM
     (128,
      dropout=0.2,
      return_sequences=True,
      return_state=True,
      recurrent_activation='relu',
      recurrent_initializer='glorot_uniform'))(lstm)

#Our model uses a bi-directional RNN, we first concatenate the hidden states from each RNN before computing the attention 
#weights and applying the weighted sum.

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
 

attention= Attention(128)
context_vector, attention_weights = attention(lstm, state_h)


 
output = keras.layers.Dense(len(set(dat1["intent"])), activation="softmax")(context_vector)
 
model = keras.Model(inputs=sequence_input, outputs=output)
 
# summarize layers
print(model.summary())


#compile model

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer="adam")

history=model.fit(X,Y, validation_split=.3, batch_size=12, epochs=125)


#predict

pdt=model.predict(X)

dat1["predicted"]=le.inverse_transform(pdt.argmax(axis=-1))

txt="what is my name?"
txt="what is ur name?"

#import numpy as np
txt=[tok.word_index[i] for i in txt]
txt=np.asarray(txt)
txt.shape
txt= txt.reshape(1,txt.shape[0])
#txt=np.zeros((1,100))
#txt=tok.texts_to_sequences(txt)
txt=pad_sequences(txt, maxlen=100)

y_p=model.predict(txt)

predicted_intent = le.inverse_transform(y_p.argmax(axis=-1))[0]


