#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:00:04 2018

@author: https://cambridgespark.com/4046-2/
"""

import nltk
nltk.download()

from nltk.corpus import brown

from gensim.models import Word2Vec
import multiprocessing

sentences= brown.sents()

EMB_DIM=300

w2v= Word2Vec(sentences, size=EMB_DIM, window=5,
              min_count=5, negative=15, iter=10, workers= multiprocessing.cpu_count())

word_vetors= w2v.wv

result= word_vetors.similar_by_word("Saturday")

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
