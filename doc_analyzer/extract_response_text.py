# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:27:42 2021

@author: ELECTROBOT
"""

#all lines
all_files={}
for file in files:
    for page in files[file]:
        for i, line in enumerate(files[file][page]):
            all_files[(file,page,i)]=line
        
        
#all pages

all_files_pages={}

for file in files:
    for page in files[file]:
        all_files_pages[(file, page)]=files[file][page]
        



all_files_text= [" ".join([" ".join([k for k in files[i][j]]) for j in files[i]]) for i in files]
all_sents=[]
[[[all_sents.append(k) for k in files[i][j]] for j in files[i]] for i in files]

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
#tf_idf_matrix
# =============================================================================
# 
# =============================================================================

question= "Fall in sales and construct  puts pressure on which community?"
question= "what is arria"
question="do you have anything on heroku and flask"
stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
question= " ".join([i for i in question.split() if i not in stopwords])

jk = vectorizer.transform([question])

from sklearn.metrics.pairwise import cosine_similarity

scores=cosine_similarity(tf_idf_matrix ,jk)
scores=[i[0] for i in scores]
dict_scores={i:j for i,j in enumerate(scores)}
dict_scores={k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse= True)}




final_responses=[all_sents[i] for i in dict_scores]

final_responses[0:3]


