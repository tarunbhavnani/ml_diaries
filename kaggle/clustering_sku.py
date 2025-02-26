# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:00:17 2024

@author: tarun
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re

# Load the data
df= pd.read_csv(r"C:\Users\tarun\Desktop\MIT-Optmization\Churn\data.csv", encoding="ISO-8859-1")


df['Description'].fillna("No Discription", inplace=True)

def clean(text):
    #convert to lower
    text= text.lower()
    #just keep words
    text=re.sub(r'\d',"", text)
    #remove extra spaces
    text=re.sub(r'\s+'," ", text)
    return text

df['Description']=df['Description'].apply(clean)

# Check the first few rows of the data
print(df.head())

# Vectorize the text descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

# Perform K-Means clustering
num_clusters = 10 # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Add the cluster labels to the original data
df['cluster'] = kmeans.labels_

# Print the top terms per cluster
print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(num_clusters):
    print(f"Cluster {i}:")
    for ind in order_centroids[i, :10]:
        print(f" {terms[ind]}")
    print("\n")

# Reduce dimensions for visualization (optional)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(tfidf_matrix.toarray())
df['pca1'] = principal_components[:, 0]
df['pca2'] = principal_components[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters of SKU Item Descriptions')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# Save the clustered data to a new CSV file
df.to_csv('clustered_sku_items.csv', index=False)






# =============================================================================
# embeddings
# =============================================================================
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load the data
#data = pd.read_csv('sku_items.csv')

# Check the first few rows of the data
#print(data.head())

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(r'D:\model_dump\bert-base-uncased')
model = BertModel.from_pretrained(r'D:\model_dump\bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply BERT embeddings to each description
data=df.head(1000)
data['embedding'] = data['Description'].apply(get_bert_embedding)

# Stack embeddings into a numpy array
embeddings = np.stack(data['embedding'].values)

# Perform K-Means clustering
num_clusters = 5  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)

# Add the cluster labels to the original data
data['cluster'] = kmeans.labels_

# Print the top terms per cluster
print("Cluster assignments:")
for i in range(num_clusters):
    print(f"Cluster {i}:")
    print(data[data['cluster'] == i]['Description'].values)
    print("\n")

# Reduce dimensions for visualization (optional)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(embeddings)
data['pca1'] = principal_components[:, 0]
data['pca2'] = principal_components[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data['pca1'], data['pca2'], c=data['cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters of SKU Item Descriptions')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# Save the clustered data to a new CSV file
data.to_csv('clustered_sku_items.csv', index=False)




