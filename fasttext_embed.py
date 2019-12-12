#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:51:30 2019

@author: tarun.bhavnani
"""

import os
os.getcwd()
os.chdir('/home/tarun.bhavnani/Desktop/dock/docker/spc')

import fasttext

# Skipgram model :
model = fasttext.train_unsupervised('data.txt', model='skipgram')
# or, cbow model :
model = fasttext.train_unsupervised('data.txt', model='cbow')


print(model.words)   # list of words in dictionary
print(model['was']) # get the vector of the word 'king'


import pandas as pd

a=pd.DataFrame()

for word in model.words:
    #print(word)
    b=pd.DataFrame([model[word]])
    b["word"]=word
    a=a.append(b)

a.index= a["word"]
a=a.drop(['word'], axis=1)