# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:31:14 2021

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

documents_df=pd.DataFrame(documents,columns=['documents'])

# removing special characters and stop words from the text
stop_words_l=stopwords.words('english')
documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )

tfidfvectoriser=TfidfVectorizer()
tfidfvectoriser.fit(documents_df.documents_cleaned)
tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)

pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T).toarray()
pairwise_differences=euclidean_distances(tfidf_vectors)

def most_similar(doc_id,similarity_matrix,matrix):
    print (f'Document: {documents_df.iloc[doc_id]["documents"]}')
    print ('\n')
    print ('Similar Documents:')
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        print('\n')
        print (f'Document: {documents_df.iloc[ix]["documents"]}')
        print (f'{matrix} : {similarity_matrix[doc_id][ix]}')

most_similar(0,pairwise_similarities,'Cosine Similarity')
most_similar(0,pairwise_differences,'Euclidean Distance')          


# =============================================================================
# w2v
# =============================================================================


# tokenize and pad every document to make them of the same size
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer()
tokenizer.fit_on_texts(documents_df.documents_cleaned)
tokenized_documents=tokenizer.texts_to_sequences(documents_df.documents_cleaned)
tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=64,padding='post')
vocab_size=len(tokenizer.word_index)+1
print (tokenized_paded_documents[0])


# loading pre-trained embeddings, each word is represented as a 300 dimensional vector
import gensim
W2V_PATH="GoogleNews-vectors-negative300.bin.gz"
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)


# creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
embedding_matrix=np.zeros((vocab_size,300))
for word,i in tokenizer.word_index.items():
    if word in model_w2v:
        embedding_matrix[i]=model_w2v[word]

# creating document-word embeddings
document_word_embeddings=np.zeros((len(tokenized_paded_documents),64,300))
for i in range(len(tokenized_paded_documents)):
    for j in range(len(tokenized_paded_documents[0])):
        document_word_embeddings[i][j]=embedding_matrix[tokenized_paded_documents[i][j]]
document_word_embeddings.shape



# calculating average of word vectors of a document weighted by tf-idf
document_embeddings=np.zeros((len(tokenized_paded_documents),300))
words=tfidfvectoriser.get_feature_names()
for i in range(len(document_word_embeddings)):
    for j in range(len(words)):
        document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i][j]
print (document_embeddings.shape)
pairwise_similarities=cosine_similarity(document_embeddings)
pairwise_differences=euclidean_distances(document_embeddings)
most_similar(0,pairwise_similarities,'Cosine Similarity')
most_similar(0,pairwise_differences,'Euclidean Distance')


# =============================================================================
# glove
# =============================================================================


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer=Tokenizer()
tokenizer.fit_on_texts(documents_df.documents_cleaned)
tokenized_documents=tokenizer.texts_to_sequences(documents_df.documents_cleaned)
tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=64,padding='post')
vocab_size=len(tokenizer.word_index)+1

# reading Glove word embeddings into a dictionary with "word" as key and values as word vectors
embeddings_index = dict()

with open(r'C:\Users\ELECTROBOT\Desktop\Python Projects\textual_entailment\glove_50\glove.6B.50d.txt', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    
# creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
embedding_matrix=np.zeros((vocab_size,50))

for word,i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
# calculating average of word vectors of a document weighted by tf-idf
document_embeddings=np.zeros((len(tokenized_paded_documents),50))
words=tfidfvectoriser.get_feature_names()

# instead of creating document-word embeddings, directly creating document embeddings
for i in range(documents_df.shape[0]):
    for j in range(len(words)):
        document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors.toarray()[i][j]
        

pairwise_similarities=cosine_similarity(document_embeddings)
pairwise_differences=euclidean_distances(document_embeddings)


most_similar(0,pairwise_similarities,'Cosine Similarity')
most_similar(0,pairwise_differences,'Euclidean Distance')          





    




