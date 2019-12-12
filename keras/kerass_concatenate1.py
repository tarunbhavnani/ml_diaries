#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:59:01 2019

@author: tarun.bhavnani
I want to concatenate the news embedding to the stock price and make predictions.
"""

#we will create a random data to amke the model
#1 is the stock prices 50 time steps
#2 is the news words
#we will concatenate them and build a model!!

import numpy as np
from keras.models import Model
n_samples = 1000
time_series_length = 50
news_words = 10
news_embedding_dim = 16
word_cardinality = 50


x_time_series = np.random.rand(n_samples, time_series_length, 1)
x_time_series.shape

x_news_words = np.random.choice(np.arange(50), replace=True, size=(n_samples, time_series_length, news_words))
x_news_words.shape
x_news_words[1]
x_news_words = [x_news_words[:, :, i] for i in range(news_words)]
x_news_words.__class__
x_news_words[1]


y = np.random.randint(2, size=(n_samples))


#define layers

## Input of normal time series
time_series_input = Input(shape=(50, 1, ), name='time_series')

## For every word we have it's own input
news_word_inputs = [Input(shape=(50, ), name='news_word_' + str(i + 1)) for i in range(news_words)]

## Shared embedding layer
from keras.layers import Embedding, Input
news_word_embedding = Embedding(word_cardinality, news_embedding_dim, input_length=time_series_length)

## Repeat this for every word position
news_words_embeddings = [news_word_embedding(inp) for inp in news_word_inputs]

## Concatenate the time series input and the embedding outputs
from keras.layers import concatenate, LSTM, Dense
concatenated_inputs = concatenate([time_series_input] + news_words_embeddings, axis=-1)

#dim: (?,50,161)
#10*16 from news and one from timeseries


## Feed into LSTM
lstm = LSTM(16)(concatenated_inputs)

## Output, in this case single classification
output = Dense(1, activation='sigmoid')(lstm)



model= Model(time_series_input, output)

model.fit([x_time_series] + x_news_words, y)


