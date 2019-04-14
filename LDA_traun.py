#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 19:35:20 2018

@author: 
#https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb
https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

"""

import os
os.getcwd()
os.chdir('/home/tarun/Desktop/LDA')
#os.chdir("/home/tarun/Desktop/gitchkdir/ml_di")

import pandas as pd
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text
documents[1:5]

len(documents.index.unique())

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


from nltk.stem import PorterStemmer as stemmer
from nltk.tokenize import sent_tokenize, word_tokenize


import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        #print(token)
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            #result.append(lemmatize_stemming(token))
            result.append(token)
    return result


doc_sample = documents[documents['index'] == 4310].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))




%time processed_docs = documents['headline_text'].map(preprocess)
%time processed_docs[:10]



#create dictionery

dictionary = gensim.corpora.Dictionary(processed_docs)
list(dictionary)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
"""
Filter out tokens that appear in

less than 15 documents (absolute number) or
more than 0.5 documents (fraction of total corpus size, not absolute number).
after the above two steps, keep only the first 100000 most frequent tokens.
"""

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
"""
Gensim doc2bow

For each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 
‘bow_corpus’, then check our selected document earlier."""

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4310]

#preview bag of words

bow_doc_4310 = bow_corpus[4310]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))


#tf-idf


from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    print(doc)
    break


#running LDA using bag of words

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, 
                                       id2word=dictionary, passes=2, workers=2)



#For each topic, we will explore the words occuring in that topic and its 
#relative weight.

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


#running LDA using tf-idf

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10,
                                             id2word=dictionary, passes=2,
                                             workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))





#performance evaluation

#bow
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
    
    
#tfidf

for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
    
#testingon unseen

unseen_document = 'How a Pentagon deal became an identity crisis for Google'
preprocess(unseen_document)
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
bow_vector
lda_model[bow_vector]
#lda model has ten topics, here it assigns the probability of the statement gping to
#each of the topics

#clearly it says thet topic 6 is the most probable.
lda_model.print_topic(6, 5)#5 is the number of top 5 terms that define the topic


"""
st=[]
for i in bow_vector:
    #print(i)
    st1=dictionary[i[0]]
    st.append(st1)
print(st)
"""
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    

