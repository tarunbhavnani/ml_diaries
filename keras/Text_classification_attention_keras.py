#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:47:19 2019

@author: tarun.bhavnani
https://androidkt.com/text-classification-using-attention-mechanism-in-keras/
"""

#prepare dataset

import os
os.chdir("/home/tarun.bhavnani/Desktop/git_tarun/ml_diaries/keras")

import tensorflow as tf
from keras_preprocessing import sequence
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate

vocab_size = 10000

pad_id = 0
start_id = 1
oov_id = 2
index_offset = 2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size, start_char=start_id,
                                                                        oov_char=oov_id, index_from=index_offset)
"""
import pickle
#save:
with open("imdb_x_train.pickle","wb") as output:
    pickle.dump(x_train,output)

with open("imdb_y_train.pickle","wb") as output:
    pickle.dump(y_train, output)

#load
with open("imdb_x_train.pickle","rb") as inp:
    ds= pickle.load(inp)
"""

word2idx = tf.keras.datasets.imdb.get_word_index()



idx2word = {v + index_offset: k for k, v in word2idx.items()}


idx2word[pad_id] = '<PAD>'
idx2word[start_id] = '<START>'
idx2word[oov_id] = '<OOV>'


print(" ".join([idx2word[i] for i in x_train[2]]))

max_len = 200
rnn_cell_size = 128

x_train = sequence.pad_sequences(x_train,
                                 maxlen=max_len,
                                 truncating='post',
                                 padding='post',
                                 value=pad_id)
x_test = sequence.pad_sequences(x_test, maxlen=max_len,
                                truncating='post',
                                padding='post',
                                value=pad_id)



#create attention layer

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






#embed layer

sequence_input = Input(shape=(max_len,), dtype='int32')

embedded_sequences = keras.layers.Embedding(vocab_size, 128, input_length=max_len)(sequence_input)


#bidirectional rnn

import os
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                     (rnn_cell_size,
                                      dropout=0.3,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_activation='relu',
                                      recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)
 
lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional \
    (tf.keras.layers.LSTM
     (rnn_cell_size,
      dropout=0.2,
      return_sequences=True,
      return_state=True,
      recurrent_activation='relu',
      recurrent_initializer='glorot_uniform'))(lstm)

#Our model uses a bi-directional RNN, we first concatenate the hidden states from each RNN before computing the attention 
#weights and applying the weighted sum.

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
 

attention= Attention(rnn_cell_size)
context_vector, attention_weights = attention(lstm, state_h)


 
output = keras.layers.Dense(1, activation='sigmoid')(context_vector)
 
model = keras.Model(inputs=sequence_input, outputs=output)
 
# summarize layers
print(model.summary())


#compile model

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

 
#early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                        min_delta=0,
#                                                        patience=1,
#                                                        verbose=0, mode='auto')



history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=200,
                    validation_split=.3, verbose=1)#, callbacks=[early_stopping_callback])


#Evaluate Model

result = model.evaluate(x_test, y_test)
print(result)


#save model
from keras.models import model_from_json
model_json= model.to_json()

with open("model_text_classify_attention_keras.json","w") as inp:
    inp.write(model_json)

model.set_weights("model_text_classify_attention_keras.h5")


# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()



