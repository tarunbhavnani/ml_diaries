#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 12:41:19 2018

@author: tarun.bhavnani@dev.smecorner.com
"""

documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"We can see the shining sun, the bright sun"
)



#covert to tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print (tfidf_matrix.shape)

#calc cosine similarity of first statement with rest of them

from sklearn.metrics.pairwise import cosine_similarity
cos_sim=cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
for i in cos_sim[0]:
  print(i)

#calculating the angles

import math
# This was already calculated on the previous step, so we just use the value
#cos_sim = 0.52305744
angle_in_radians = [math.acos(i) for i in cos_sim[0]]
angle_in_degrees = [math.degrees(math.acos(i)) for i in cos_sim[0]]


