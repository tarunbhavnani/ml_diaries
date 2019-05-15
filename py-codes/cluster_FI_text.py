#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:42:10 2018

@author: tarun.bhavnani@dev.smecorner.com
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:16:39 2018

@author: tarun.bhavnani@dev.smecorner.com
"""



import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize 
from nltk.util import ngrams
from collections import Counter



"############INPUT file###########################"

dat=pd.read_excel("office FI_full_out.xlsx")
dat.columns=("Remarks","app_id")
num_clusters = 3
"##################################################"





#dat=pd.read_excel("office FI_full_out.xlsx")


titles=dat.app_id
synopses=dat.Remarks


###############################code############333333
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
totalvocab_stemmed = []
totalvocab_tokenized = []
#i=synopses.iloc[1]
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) 
    totalvocab_stemmed.extend(allwords_stemmed) 
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)       
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

tfidf_vectorizer = TfidfVectorizer(max_df=.8, max_features=200000,
                                 min_df=0.2, stop_words=stopwords,
                                 use_idf=True,
                                 ngram_range=(3,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) 

tmdf = pd.DataFrame(tfidf_matrix.toarray())
terms = tfidf_vectorizer.get_feature_names()
#num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
frame=pd.DataFrame(list(zip(titles, synopses, clusters)),columns = [ 'title','synopses' ,'cluster'])

##