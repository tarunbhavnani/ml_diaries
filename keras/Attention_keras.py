#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:59:25 2019

@author: tarun.bhavnani
"""

import os
os.chdir("/home/tarun.bhavnani/Desktop/git_tarun/attention_keras")

batch_size=32
en_timesteps=20
en_vsize=2000
#inputs
from keras.layers import Input
encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize),
                       name='encoder_inputs')
decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize),
                       name='decoder_inputs')


#define encoder
from keras.layers import GRU
encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True,
                  name='encoder_gru')
encoder_out, encoder_state = encoder_gru(encoder_inputs)


#define decoder

decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True,
                  name='decoder_gru')
decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)



#define attention
os.chdir('/home/tarun.bhavnani/Desktop/git_tarun')
from attention_keras.layers.attention import AttentionLayer

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_out, decoder_out])


from keras.layers import Concatenate
#Concatenate the attn_out and decoder_out as an input to the softmax layer.
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])


##Concatenate the attn_out and decoder_out as an input to the softmax layer.
#decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

#finally
#Define TimeDistributed Softmax layer and provide decoder_concat_input as the input.

from keras.layers import Dense,TimeDistributed

dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
dense_time = TimeDistributed(dense, name='time_distributed_layer')
decoder_pred = dense_time(decoder_concat_input)


#define final model

full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
full_model.compile(optimizer='adam', loss='categorical_crossentropy')


