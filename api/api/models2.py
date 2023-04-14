# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:59:50 2021

@author: ELECTROBOT
"""
#data
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
docs= twenty_train.data
target= twenty_train.target


from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(docs)
#count_vect.vocabulary_.get('algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB().fit(X_train_tf, target)


#get pipeline
from sklearn.pipeline import Pipeline
text_clf= Pipeline([('vect', CountVectorizer()),
                    ('tfdif_transformer',TfidfTransformer() ),
                    ('clf', MultinomialNB())
    ])

text_clf.fit(docs, target)


twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
actuals = twenty_test.target

preds= text_clf.predict(docs_test)

from sklearn import metrics

cr=metrics.classification_report(actuals, preds)

# =============================================================================

import pickle
pickle.dump(text_clf, open('models/final_prediction_sk.pickle', 'wb'))

# =============================================================================
import numpy as np
modelfile='./models/final_prediction_sk.pickle'
model= np.load(open(modelfile,'rb'),allow_pickle=True)

#model.predict(docs_test)
text="From  rind enterprise bih harvard edu  David Rind  Subject  Re  Candida yeast  Bloom  Fact or Fiction Organization  Beth Israel Hospital  Harvard Medical School  Boston Mass   USA Lines     NNTP Posting Host  enterprise bih harvard edu  In article      Apr            vms ocom okstate edu   banschbach vms ocom okstate edu writes   are in a different class"
model.predict([text]).item()










