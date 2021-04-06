# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:46:27 2021

@author: ELECTROBOT
"""

import re

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

from sklearn.feature_extraction.text import TfidfVectorizer

#company_names = names['Company Name']
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
vectorizer.fit(all_sents)
tf_idf_matrix= vectorizer.transform(all_sents)
tf_idf_matrix

jk = vectorizer.transform(["federer is married to whome"])

from sklearn.metrics.pairwise import cosine_similarity

scores=cosine_similarity(tf_idf_matrix ,jk)
scores=[i[0] for i in scores]
dict_scores={i:j for i,j in enumerate(scores)}
dict_scores={k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse= True)}




[all_sents[i] for i in dict_scores]
