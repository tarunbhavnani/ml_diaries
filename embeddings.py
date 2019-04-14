#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:00:04 2018

@author: https://cambridgespark.com/4046-2/

Creating a w2vec embedding out of a set of docs

"""

import nltk
#nltk.download()

from nltk.corpus import brown

#!pip install gensim
from gensim.models import Word2Vec
import multiprocessing

sentences= brown.sents()
#normally a list of docs will be like this
sentences=[" ".join([j for j in i]) for i in sentences]
sentences=[i.lower() for i in sentences]
import re
sentences= [re.sub("[^a-z\s]","",x) for x in sentences]

#now we will prepare
sentences=[i.split() for i in sentences]


EMB_DIM=300

w2v= Word2Vec(sentences, size=EMB_DIM, window=5,
              min_count=5, negative=15, iter=10, workers= multiprocessing.cpu_count())

word_vetors= w2v.wv

result= word_vetors.similar_by_word("saturday")

from nltk.corpus import con112000
nltk.corpus.treebank.tagged_words()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
import collections

train_words= nltk.corpus.treebank.tagged_words()
train_words[:20]
