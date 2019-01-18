#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:55:20 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

import numpy as np
import pandas as pd
import nltk
import re
import os
os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/exploration/new/field_extract')
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
dat=dat.drop_duplicates(["app_id"])
dat=dat.drop_duplicates(["Remarks"])
#target=pd.read_excel("target.xlsx")
dat=dat[dat.app_id.notna()]

remarks= dat.Remarks
app_id= dat.app_id

remarks= [x.lower() for x in remarks]
#remarks= [re.sub("(*a-z)","",x) for x in remarks]==how to remove all bet brackets
remarks= [re.sub("[^a-z\s]","",x) for x in remarks]
"can decide to use numbers as well since rent and dates etc get deleted"

#remarks=[re.sub("[^A-Za-z0-9\s]","",x) for x in remarks]

"converting sentences to list of words"
fin=[]
for i in remarks:
  f1=[]
  for j in i.split():
    f1.append(j)
  fin.append(f1)


from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(fin)
dict_fin=dict([(x[0],x[1]) for x in dictionary.iteritems()])
dict_fin[1]#also
rev_dict_fin=dict([(x[1],x[0]) for x in dictionary.iteritems()])
rev_dict_fin["also"]#1


#lots fo spelling mistakes, we can use character level model!!
from gensim.models import Word2Vec

model=Word2Vec(fin,min_count=1)
#model["also"]
model.most_similar("nyasa")
len(list(model.wv.vocab))
model.wv.get_vector("also")

"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer= Tokenizer()
tokenizer.fit_on_texts(texts=remarks)
seq= tokenizer.texts_to_sequences(remarks)

hj=tokenizer.sequences_to_matrix(seq)#BOW
msl= max([len(x) for x in seq])
seq_pad= pad_sequences(seq, padding="post")


rev_idx=dict([(tokenizer.word_index[i],i) for i  in tokenizer.word_index])
rev_idx[0]="unk"
fin=[]
for i in seq:
  f1=[]
  for j in i:
    f1.append(rev_idx[j])
  fin.append(f1)

"""

#########################################################################################3333

"""
#the explaination of sent_vectorizer
we have seen man+king=women=queen
word embeddings added gives the meaning of the whole thing put together.
sent_vectorizer adds the word embeddings of all the words of a sentence
dimensions remain same but values change.

This will thus make more and more sense if all the faaltu words are removed much before
"""



def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw




X=[]
for sentence in fin:
    X.append(sent_vectorizer(sentence, model))   


X[1].shape
"each element of X represents the nth sentences , sum of all he word vectors it has."

from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics

NUM_CLUSTERS=3

kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)

assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

#from collections import Counter
Counter(assigned_clusters)

df1= pd.DataFrame(data= list(zip(app_id, assigned_clusters)), columns=["app_id", "clusters"])

#we can try to visualize statements on graph usinh T-sne



"##################################################"
#use pretrained glove model
path="/home/tarun.bhavnani@dev.smecorner.com/Desktop/Embedding Data"
file="/glove.6B.100d.txt.word2vec"
model_2 = Word2Vec(size=100, min_count=1)
model_2.build_vocab(fin)
model_2.corpus_count


X1=[]
for sentence in fin:
    X1.append(sent_vectorizer(sentence, model_2))   


NUM_CLUSTERS=3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X1, assign_clusters=True)
#from collections import Counter
Counter(assigned_clusters)

df2= pd.DataFrame(data= list(zip(app_id, assigned_clusters)), columns=["app_id", "clusters"])



"##############################################################"


#Topic analysis###############################################################################################3

from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(fin)
dict_fin=dict([(x[0],x[1]) for x in dictionary.iteritems()])
dict_fin[1]#also
rev_dict_fin=dict([(x[1],x[0]) for x in dictionary.iteritems()])
rev_dict_fin["also"]#1

stopwords=["and","a","the","is","then","of","an","that","by","to","n","as","on","us","he","said","this","we","at", "business", "seen", "applicant"]
fin_rm = [[x for x in text if x not in stopwords] for text in fin] 
common_corpus = [dictionary.doc2bow(x) for x in fin_rm]
len(common_corpus)#number of documents
#print(fin[1])
#print([(dict_fin[i[0]],i[1]) for i in common_corpus[1]])

#create bow from this, jlt!
bow=np.zeros((len(fin), len(dict_fin)))

for i,j in enumerate(common_corpus):
  #print(i)
  for k in j:
#    print(j)
    bow[i,k[0]]=k[1]



# Train the model on the corpus.##########################################3

from gensim.models.ldamodel import LdaModel
lda = LdaModel(common_corpus, num_topics=10,id2word=dictionary)


#list(lda)
lda.print_topics()



"################################################################
"same topics seen in earlier"
"we remove the words which are seen in more than 500 docs and also words which are there in less than 10 docs"
"and do the same exercise again"
word_appear_docs={}

for text in fin:
  #print(text)
  for word in set(text):
    #print(word)
    if word in word_appear_docs:
      word_appear_docs[word]+=1
    else:
      word_appear_docs[word]=1

max(word_appear_docs.values())
min(word_appear_docs.values())
import matplotlib.pyplot as plt
plt.plot(word_appear_docs.values())

word_list=[x for x in word_appear_docs if word_appear_docs[x]>10 and word_appear_docs[x]<500]


fin_rm_words = [[x for x in text if x  in word_list] for text in fin] 
common_corpus = [dictionary.doc2bow(x) for x in fin_rm_words]
len(common_corpus)#number of documents

# Train the model on the corpus.##########################################3
lda = LdaModel(common_corpus, num_topics=3,id2word=dictionary)


#list(lda)
lda.print_topics()

#sorting the word appear docs dictionery to see which worsd are appearing in most/least docs
import operator
#x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
#sorted_x = sorted(x.items(), key=operator.itemgetter(1))
sorted_x = sorted(word_appear_docs.items(), key=operator.itemgetter(1), reverse=True)

#or
d=word_appear_docs

d=sorted(d.items(), key=lambda x: x[1])

######################################################################################3



