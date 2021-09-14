# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 12:03:32 2021

@author: tarun
"""
# Extract word vectors

import numpy as np
word_embeddings = {}
f = open(r'C:\Users\tarun\Desktop\summarization_bart\glove\glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()



# remove punctuations, numbers and special characters
clean_sentences = pd.Series(all_sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


from nltk.corpus import stopwords
stop_words = stopwords.words('english')

 
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
  
# similarity matrix
sim_mat = np.zeros([len(all_sentences), len(all_sentences)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(all_sentences)):
  for j in range(len(all_sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
      




import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)




ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(all_sentences)), reverse=True)
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])


