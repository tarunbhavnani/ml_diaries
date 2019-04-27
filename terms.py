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

imbalanced data--propensity to buy at tcs, and rest of tcs projects

svm: Separation of classes. That’s what SVM does. Kernels(linera, polynomial) Polynomial and exponential kernels calculates separation line in higher dimension. This is called kernel trick
gamma: low gamma considers points close to plane, high gamma considers farther points
C: big c low margin, low C high margin

svc
cnn
lstm
bilstm
nn
rnn
optimizers
batch normalization
gd/sgd/adam/rmsprop/adagrad

randomforest
dt
xgboost

seq2seq, return sequences, return state
nmt
autoencoders

dropout

confusion matrix

lift--The basic idea of lift analysis is as follows:

group data based on the predicted churn probability (value between 0.0 and 1.0). Typically, you look at deciles, so you'd have 10 groups: 0.0 - 0.1, 0.1 - 0.2, ..., 0.9 - 1.0
calculate the true churn rate per group. That is, you count how many people in each group churned and divide this by the total number of customers per group.







Rasa NLU
The two components between which you can choose are:

Pretrained Embeddings (Intent_classifier_sklearn)
Supervised Embeddings (Intent_classifier_tensorflow_embedding)

Word embeddings are vector representations of words, meaning each word is converted to a dense numeric vector. Word embeddings capture semantic and syntactic aspects of words. This means that similar words should be represented by similar vectors.

intent_classifier_tensorflow_embedding--t trains word embeddings from scratch. It is typically used with the intent_featurizer_count_vectors component which counts how often distinct words of your training data appear in a message and provides that as input for the intent classifier.


Extracting Entities
ner_spacy: pre-traines. Entity recognition with SpaCy language models

ner_http_duckling: Rule based entity recognition using Facebook’s Duckling: -->amounts of money, dates, distances, or durations

ner_crf: raining an extractor for custom entities: --> Neither ner_spacy nor ner_duckling require you to annotate any of your training data, since they are either using pretrained classifiers (spaCy) or rule-based approaches (Duckling). The ner_crf component trains a conditional random field which is then used to tag entities in the user messages. Since this component is trained from scratch as part of the NLU pipeline you have to annotate your training data yourself. This is an example from our documentation on how to do so
also has regex support and lookup tables

ner_synonyms


generative and discriminative: https://en.wikipedia.org/wiki/Discriminative_model
The typical discriminative learning approaches include Logistic Regression (LR), Support Vector Machine (SVM), conditional random fields (CRFs) (specified over an undirected graph), and others. The typical generative model approaches contain Naive Bayes, Gaussian Mixture Model, and others.



#sklearn  feature on text
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.category_id
features.shape

#correlated terms with each category and more... full code on:
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f



precision: TP/TP+FP
recall:    TP/TP+FN
f1: harmonic mean of precision and recall


Imbalanced data: One of the tactics of combating imbalanced classes is using Decision Tree algorithms, so, we are using Random Forest classifier to learn imbalanced data and set class_weight=balanced 


pca--> 400 images 64*64---> 400 vectors of 4096 elements--> 400 pca
Each of the 400 original images (i.e. each of the 400 original rows of the matrix) can be expressed as a (linear) combination of the 400 pca's.
The goal of PCA is to reveal typical vectors: each of the creepy/typical guy represents one specific aspect underlying the data.
each of the PCA's captures a specific aspect of the data. Each principal component captures a specific latent factor.


Matrix--> users on rows and movies on cols
PCA on R=M
PCA on R(T)---> PCA on Matrix--> movies on rows and users on cols--> U(T)

SVD on this R matrix will give--> MU(T) or MEU(T) where E is a diagonal matrix
basically what it does is:

rui  =  pu⋅qi  =  ∑f∈latent factorsaffinity of u for f×affinity of i for f
affinity of user for f into affinity of movie for f




#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

Neural Network
A neural network usually involves a large number of processors operating in parallel and arranged in tiers. The first tier receives the raw input information -- analogous to optic nerves in human visual processing. Each successive tier receives the output from the tier preceding it, rather than from the raw input -- in the same way neurons further from the optic nerve receive signals from those closer to it. The last tier produces the output of the system.


RNN


LSTM


CNN


Autoencoder, encoder/decoder


GAN


Word Embedding


Word2vec


Seq2Seq


Optimizer

Adam
rmsprop

Gradient Descent

SGD

SVM

SVD

PCA

Random Forest

Logistic Regression

Linera Regression

Decision Tree


Clustering
KNN

Kmeans


Parametric/Non Parametric


Supervided/Non Supervised





