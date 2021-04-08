# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:35:27 2021

@author: ELECTROBOT
"""

#text classification functions

import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout, LSTM, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



def simple_clean(sent):
    sent= re.sub(r'[^A-Za-z0-9 /.,]', " ", sent.lower())
    sent= re.sub(r'\s+', " ", sent)
    return sent

def glove_embeddings(path_to_glove, get_embedding_matrix=False, tok=None):
    
    embeddings_index = dict()

    f = open(path_to_glove)
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    if get_embedding_matrix and tok:
        
        # create a weight matrix for words in training docs
        vocab_size= len(tok.word_index)+1
        embedding_matrix = np.zeros((vocab_size, 50))
        for word, i in tok.word_index.items():
        	embedding_vector = embeddings_index.get(word)
        	if embedding_vector is not None:
        		embedding_matrix[i] = embedding_vector
                
        return embedding_matrix
    else:
        return embeddings_index


def model_cnn(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model



def tfidf_model():
    


