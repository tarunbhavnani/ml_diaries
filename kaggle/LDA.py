# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:08:04 2023

@author: tarun
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# Vectorize the text data using a bag-of-words representation
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

# Fit an LDA model to the data
lda = LatentDirichletAllocation(n_components=20, random_state=42)
lda.fit(X)

# Print the top 10 words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))
    print()
    



#which toipic for which document

# Get the topic distribution for each document
doc_topic_dist = lda.transform(X)

# Print the top topic for the first 10 documents
df= pd.DataFrame({"text":newsgroups.data, "label":newsgroups.target})

df["lda"]=999

for i in range(len(df)):
    df["lda"].iloc[i] = doc_topic_dist[i].argmax()
    print("Document %d: Top Topic = %d" % (i, top_topic_idx))


df.lda.value_counts()
df.label.value_counts()


#pca
import numpy as np
from sklearn.decomposition import PCA

# Generate some sample data
X = np.random.randn(100, 5)  # 100 samples with 5 features each

# Create a PCA object with 2 components
pca = PCA(n_components=4)

# Fit the PCA model to the data
pca.fit(X)

# Transform the data to the lower-dimensional space
X_pca = pca.transform(X)

# Print the explained variance ratio of each component
print(pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_)


# Access the components (i.e. eigenvectors) of the PCA model
components = pca.components_

# Print the components
print(components)