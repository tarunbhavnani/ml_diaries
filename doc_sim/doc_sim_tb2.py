# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:27:07 2021

@author: ELECTROBOT
"""
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# Sample corpus
documents = ['Machine learning is the study of computer algorithms that improve automatically through experience.\
Machine learning algorithms build a mathematical model based on sample data, known as training data.\
The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
where no fully satisfactory algorithm is available.',
'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
It involves computers learning from data provided so that they carry out certain tasks.',
'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
'Software engineering is the systematic application of engineering approaches to the development of software.\
Software engineering is a computing discipline.',
'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
Developing a machine learning application is more iterative and explorative process than software engineering.'
]

#documents_df=pd.DataFrame(documents,columns=['documents'])
stop_words_l=stopwords.words('english')

documents=[re.sub(r'[^a-zA-Z ]',"",i.lower()) for i in documents]
documents=[" ".join([j for j in i.split() if j not in stop_words_l]) for i in documents]



# =============================================================================
# tokenize 
# =============================================================================

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen=64

tok= Tokenizer(split=" ")
tok.fit_on_texts(documents)
tokenized_document=tok.texts_to_sequences(documents)
tokenized_padded_document= pad_sequences(tokenized_document, maxlen=maxlen, padding='post')

vocab= tok.word_index
vocab_inv= {i:j for j,i in vocab.items()}
num_vocab= len(tok.word_index)+1

# =============================================================================
# read glove
# =============================================================================
# reading Glove word embeddings into a dictionary with "word" as key and values as word vectors
embeddings_index = dict()

with open(r'C:\Users\ELECTROBOT\Desktop\Python Projects\textual_entailment\glove_50\glove.6B.50d.txt', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
embedding_matrix=np.zeros((num_vocab,50))

for word,i in tok.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# =============================================================================
# document vectors or sentence vectors
# =============================================================================
#we can directly mean or average them but this will be a little crude
#we can multiply each word vec by the respective tfidf value.


#create tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer()
tfidf.fit(documents)
tfidf_vectors= tfidf.transform(documents)

df_tfidf=pd.DataFrame.sparse.from_spmatrix(tfidf_vectors)
df_tfidf.columns=tfidf.get_feature_names()



    


# =============================================================================
# conver docs to embeddings
# =============================================================================



document_embeddings= np.zeros((len(documents), 50))


for doc_num,docu in enumerate(documents):
    print(doc_num)
    for word in vocab:
        #print(word)
        try:
            document_embeddings[doc_num]+=embedding_matrix[tok.word_index[word]]*df_tfidf.iloc[doc_num][word]
        except:
            print(word)
            document_embeddings[doc_num]+=np.zeros(50)
        



























