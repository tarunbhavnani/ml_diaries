#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:13:18 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re
import os
os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/keras')
df = pd.read_csv("tennis_articles_v4.csv")
df.iloc[0]
df['article_text'][0]

from nltk.tokenize import sent_tokenize

sentences=[]
for s in df["article_text"]:
  #print(s)
  sentences.append(sent_tokenize(s))

#flatten list
sentences = [y for x in sentences for y in x] # flatten list

sentences[:5]

#get glove
# Extract word vectors
word_embeddings = {}
f = open('glove.6B.50d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


len(word_embeddings)


# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


sentence_vectors = [] #sum of embeddings for all the worsd1!
for i in clean_sentences:
  #print(i)
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((50,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((50,))
  sentence_vectors.append(v)
  
  
# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0,0]


import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)



ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])


