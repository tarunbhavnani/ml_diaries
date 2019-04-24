#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:35:53 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
rasa core and nlu deep learning models, how they train on the data
ner-everything:classify named entities in text into pre-defined categories such as the names of persons, 
organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

can use nltk
nltk.word_tokenize, pos_tag, etc
oe spacy:
  nlp=en_core_web_sm.load()





relation extraction-- same as above

text summary
text classification


lda-- is a generative probabilistic model, that assumes a Dirichlet prior over the latent topics..lda finds the latent topics in the documents. it tries to make a words*w, w*docs such a way 
to minimize the error. we have different topics which are clusters of diff words and each document 
has a propb of fallinf into each one of topics.

lsa- its more like PCA.learns latent topics by performing a matrix decomposition (SVD) 
on the term-document matrix.

lsa is faster but lesser accuracy mostly.

pca:Principal components analysis is a procedure for identifying a smaller number of uncorrelated variables, called “principal components”, from a large set of data. The goal of principal components analysis is to explain the maximum amount of variance with the fewest number of principal components.


cnn
lstm
bilstm
nn
rnn
optimizers
batch normalization
gd/sgd/adam/rmsprop/adagrad

randomforest
svm
svc
dt
xgboost

seq2seq, return sequences, return state
nmt
autoencoders

dropout

confusion matrix
tpr fpr f1 score

lift

