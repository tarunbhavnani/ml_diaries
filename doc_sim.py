# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:30:12 2021

@author: ELECTROBOT
"""



import re
import string
documents_clean = []
for d in documents:
    # Remove Unicode
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
    # Remove Mentions
    document_test = re.sub(r'@\w+', '', document_test)
    # Lowercase the document
    document_test = document_test.lower()
    # Remove punctuations
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    # Lowercase the numbers
    document_test = re.sub(r'[0-9]', '', document_test)
    # Remove the doubled space
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    document_test = re.sub(r'\n', ' ', document_test)
    documents_clean.append(document_test)
    



from sklearn.feature_extraction.text import TfidfVectorizer
# Instantiate a TfidfVectorizer object
vectorizer = TfidfVectorizer()
# It fits the data and transform it as a vector
X = vectorizer.fit_transform(documents_clean)
# Convert the X as transposed matrix
X = X.T.toarray()
# Create a DataFrame and set the vocabulary as the index
df = pd.DataFrame(X, index=vectorizer.get_feature_names())
df.head()
len(documents_clean)
df.shape


import numpy as np

def get_similar_articles(q, df):
  print("query:", q)
  print("Berikut artikel dengan nilai cosine similarity tertinggi: ")
  # Convert the query become a vector
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
  sim = {}
  # Calculate the similarity
  for i in range(len(documents_clean)):
    sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
  
  # Sort the values 
  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  # Print the articles and their similarity values
  for k, v in sim_sorted[0:10]:
    if v != 0.0:
      print("Nilai Similaritas:", v)
      print(documents_clean[k])
      print()



# Add The Query
q = 'legitimate proposal '
# Call the function
get_similar_articles(q1, df)