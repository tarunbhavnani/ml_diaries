#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:28:43 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



vectorizer1 = CountVectorizer()
vectorizer2 = TfidfVectorizer()


text=["tarun bhavnani","works in smecorner","is a data dcientist and an NLP scientost","tarun is building a chatbot"]

tt1=vectorizer1.fit_transform(text)
vectorizer1.vocabulary_
tt2=vectorizer2.fit_transform(text)
vectorizer2.vocabulary_


print(tt2)
tt1.toarray()
tt2.toarray()


vectorizer3 = CountVectorizer(ngram_range=(1, 2))
vectorizer4 = TfidfVectorizer(ngram_range=(1, 2))

tt3=vectorizer3.fit_transform(text)
vectorizer3.vocabulary_
tt4=vectorizer4.fit_transform(text)
vectorizer4.vocabulary_
