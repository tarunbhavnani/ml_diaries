# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:07:43 2024

@author: tarun
"""

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

import json
import pandas as pd

file_name=r"D:\kaggle\arxiv\arxiv.json"


cols = ['id', 'title', 'abstract', 'categories']
data = []

with open(file_name, encoding='latin-1') as f:
    for line in f:
        doc = json.loads(line)
        lst = [doc['id'], doc['title'], doc['abstract'], doc['categories']]
        data.append(lst)

df = pd.DataFrame(data=data, columns=cols)#.sample(n=10_000, random_state=68)

df.head()

category_counts = df['categories'].value_counts()

filtered_categories = category_counts[category_counts > 1000].index.tolist()

filtered_df = df[df['categories'].isin(filtered_categories)]

filtered_df['cat']=[i.split(".")[0] for i in filtered_df['categories']]

filtered_df['categories'].value_counts()
filtered_df['cat'].value_counts()


#select 5 pc of stratified random data
fdf=filtered_df.groupby('cat').apply(lambda x: x.sample(frac=.05))
fdf.cat.value_counts()




# =============================================================================
# 
# =============================================================================

model= r"D:\model_dump\bert-base-uncased"
def create_sentence_embeddings(sentences, model, batch_size=32):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model)
    model = BertModel.from_pretrained(model)
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize an empty array to store embeddings
    embeddings_list = []

    # Process sentences in batches
    for  i in range(0, len(sentences), batch_size):
      
        print(".", end=" ", flush=True)
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize and encode the sentences
        encoded_input = tokenizer(batch_sentences, padding=True, max_length=512,truncation=True, return_tensors='pt').to(device)

        # Forward pass through the BERT model
        with torch.no_grad():
            outputs = model(**encoded_input)

        # Extract the embeddings from the output
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Using mean pooling
        embeddings_list.append(embeddings)

    # Concatenate embeddings from all batches
    all_embeddings = np.concatenate(embeddings_list, axis=0)

    return all_embeddings



# Create sentence embeddings for all sentences
all_embeddings = create_sentence_embeddings(list(fdf.abstract), model)

#slow



#get tfidf embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


# Load a sample dataset for illustration purposes

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.75, min_df=5, stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(fdf.abstract)

from sklearn.decomposition import TruncatedSVD

# Apply Truncated SVD for dimensionality reduction
num_topics = len(set(fdf.cat)) # Specify the number of topics (adjust as needed)
svd_model = TruncatedSVD(n_components=num_topics, random_state=42)

svd_topic_matrix = svd_model.fit_transform(tfidf_matrix)


# =============================================================================
# Topic similarities 
# =============================================================================



topic_similarities = cosine_similarity(all_embeddings)
#topic_similarities = cosine_similarity(tfidf_matrix)


indx=1
text=filtered_df.abstract.iloc[indx]

#similar text
top_n=10

similar_text = topic_similarities[indx].argsort()[:-top_n-1:-1]

recommendations = filtered_df.iloc[similar_text]['categories'].tolist()






# =============================================================================
# clusters visualize by umap
# =============================================================================

replacements = {'astro-ph': 'Physics',
 'q-alg': 'Physics',
 'cond-mat': 'Physics',
 'chao-dyn': 'Mathematics',
 'chem-ph': 'Physics',
 'plasm-ph': 'Physics',
 'alg-geom': 'Mathematics',
 'cmp-lg': 'Computer Science',
 'patt-sol': 'Physics',
 'dg-ga': 'Mathematics',
 'funct-an': 'Mathematics',
 'mtrl-th': 'Physics',
 'cs.AI': 'Computer Science',
 'cs.AR': 'Computer Science',
 'cs.CC': 'Computer Science',
 'cs.CE': 'Computer Science',
 'cs.CG': 'Computer Science',
 'cs.CL': 'Computer Science',
 'cs.CR': 'Computer Science',
 'cs.CV': 'Computer Science',
 'cs.CY': 'Computer Science',
 'cs.DB': 'Computer Science',
 'cs.DC': 'Computer Science',
 'cs.DL': 'Computer Science',
 'cs.DM': 'Computer Science',
 'cs.DS': 'Computer Science',
 'cs.ET': 'Computer Science',
 'cs.FL': 'Computer Science',
 'cs.GL': 'Computer Science',
 'cs.GR': 'Computer Science',
 'cs.GT': 'Computer Science',
 'cs.HC': 'Computer Science',
 'cs.IR': 'Computer Science',
 'cs.IT': 'Computer Science',
 'cs.LG': 'Computer Science',
 'cs.LO': 'Computer Science',
 'cs.MA': 'Computer Science',
 'cs.MM': 'Computer Science',
 'cs.MS': 'Computer Science',
 'cs.NA': 'Computer Science',
 'cs.NE': 'Computer Science',
 'cs.NI': 'Computer Science',
 'cs.OH': 'Computer Science',
 'cs.OS': 'Computer Science',
 'cs.PF': 'Computer Science',
 'cs.PL': 'Computer Science',
 'cs.RO': 'Computer Science',
 'cs.SC': 'Computer Science',
 'cs.SD': 'Computer Science',
 'cs.SE': 'Computer Science',
 'cs.SI': 'Computer Science',
 'cs.SY': 'Computer Science',
 'econ.EM': 'Economics',
 'econ.GN': 'Economics',
 'econ.TH': 'Economics',
 'eess.AS': 'Electrical Engineering and Systems Science',
 'eess.IV': 'Electrical Engineering and Systems Science',
 'eess.SP': 'Electrical Engineering and Systems Science',
 'eess.SY': 'Electrical Engineering and Systems Science',
 'math.AC': 'Mathematics',
 'math.AG': 'Mathematics',
 'math.AP': 'Mathematics',
 'math.AT': 'Mathematics',
 'math.CA': 'Mathematics',
 'math.CO': 'Mathematics',
 'math.CT': 'Mathematics',
 'math.CV': 'Mathematics',
 'math.DG': 'Mathematics',
 'math.DS': 'Mathematics',
 'math.FA': 'Mathematics',
 'math.GM': 'Mathematics',
 'math.GN': 'Mathematics',
 'math.GR': 'Mathematics',
 'math.GT': 'Mathematics',
 'math.HO': 'Mathematics',
 'math.IT': 'Mathematics',
 'math.KT': 'Mathematics',
 'math.LO': 'Mathematics',
 'math.MG': 'Mathematics',
 'math.MP': 'Mathematics',
 'math.NA': 'Mathematics',
 'math.NT': 'Mathematics',
 'math.OA': 'Mathematics',
 'math.OC': 'Mathematics',
 'math.PR': 'Mathematics',
 'math.QA': 'Mathematics',
 'math.RA': 'Mathematics',
 'math.RT': 'Mathematics',
 'math.SG': 'Mathematics',
 'math.SP': 'Mathematics',
 'math.ST': 'Mathematics',
 'astro-ph.CO': 'Physics',
 'astro-ph.EP': 'Physics',
 'astro-ph.GA': 'Physics',
 'astro-ph.HE': 'Physics',
 'astro-ph.IM': 'Physics',
 'astro-ph.SR': 'Physics',
 'cond-mat.dis-nn': 'Physics',
 'cond-mat.mes-hall': 'Physics',
 'cond-mat.mtrl-sci': 'Physics',
 'cond-mat.other': 'Physics',
 'cond-mat.quant-gas': 'Physics',
 'cond-mat.soft': 'Physics',
 'cond-mat.stat-mech': 'Physics',
 'cond-mat.str-el': 'Physics',
 'cond-mat.supr-con': 'Physics',
 'gr-qc': 'Physics',
 'hep-ex': 'Physics',
 'hep-lat': 'Physics',
 'hep-ph': 'Physics',
 'hep-th': 'Physics',
 'math-ph': 'Physics',
 'nlin.AO': 'Physics',
 'nlin.CD': 'Physics',
 'nlin.CG': 'Physics',
 'nlin.PS': 'Physics',
 'nlin.SI': 'Physics',
 'nucl-ex': 'Physics',
 'nucl-th': 'Physics',
 'physics.acc-ph': 'Physics',
 'physics.ao-ph': 'Physics',
 'physics.app-ph': 'Physics',
 'physics.atm-clus': 'Physics',
 'physics.atom-ph': 'Physics',
 'physics.bio-ph': 'Physics',
 'physics.chem-ph': 'Physics',
 'physics.class-ph': 'Physics',
 'physics.comp-ph': 'Physics',
 'physics.data-an': 'Physics',
 'physics.ed-ph': 'Physics',
 'physics.flu-dyn': 'Physics',
 'physics.gen-ph': 'Physics',
 'physics.geo-ph': 'Physics',
 'physics.hist-ph': 'Physics',
 'physics.ins-det': 'Physics',
 'physics.med-ph': 'Physics',
 'physics.optics': 'Physics',
 'physics.plasm-ph': 'Physics',
 'physics.pop-ph': 'Physics',
 'physics.soc-ph': 'Physics',
 'physics.space-ph': 'Physics',
 'quant-ph': 'Physics',
 'q-bio.BM': 'Quantitative Biology',
 'q-bio.CB': 'Quantitative Biology',
 'q-bio.GN': 'Quantitative Biology',
 'q-bio.MN': 'Quantitative Biology',
 'q-bio.NC': 'Quantitative Biology',
 'q-bio.OT': 'Quantitative Biology',
 'q-bio.PE': 'Quantitative Biology',
 'q-bio.QM': 'Quantitative Biology',
 'q-bio.SC': 'Quantitative Biology',
 'q-bio.TO': 'Quantitative Biology',
 'q-fin.CP': 'Quantitative Finance',
 'q-fin.EC': 'Quantitative Finance',
 'q-fin.GN': 'Quantitative Finance',
 'q-fin.MF': 'Quantitative Finance',
 'q-fin.PM': 'Quantitative Finance',
 'q-fin.PR': 'Quantitative Finance',
 'q-fin.RM': 'Quantitative Finance',
 'q-fin.ST': 'Quantitative Finance',
 'q-fin.TR': 'Quantitative Finance',
 'stat.AP': 'Statistics',
 'stat.CO': 'Statistics',
 'stat.ME': 'Statistics',
 'stat.ML': 'Statistics',
 'stat.OT': 'Statistics',
 'stat.TH': 'Statistics'}

filtered_df.categories.apply(replacements)
filtered_df['group_name']=[replacements[i] if i in replacements else i for i in filtered_df.categories]




#import umap
import umap.umap_ as umap
reducer = umap.UMAP(n_neighbors=5,
                    n_components=2,
                    n_jobs = -1)

embedding_2d = reducer.fit_transform(all_embeddings)
filtered_df["abstract_vector_x"] = embedding_2d[:, 0]
filtered_df["abstract_vector_y"] = embedding_2d[:, 1]



fdf= filtered_df[filtered_df['group_name'].isin(['Statistics','Quantitative Finance','Physics','Electrical Engineering and Systems Science', 'Economics','Computer Science','Mathematics'])]

import seaborn as sns

sns.set(context="paper", style="white")
sns.set(rc = {'figure.figsize' : (12, 12)})

ax = sns.scatterplot(data=fdf,
                x="abstract_vector_x",
                y="abstract_vector_y",
                hue="group_name")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# =============================================================================
# lets cluster
# first we will try to find how many clusters
# once we decide we will do the clusterung
# then we will plot
# =============================================================================



#kmeans

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming 'all_embeddings' is your matrix of document embeddings
# 'all_embeddings' should have shape (number_of_documents, embedding_dimension)

# Example: Generate random embeddings (replace this with your actual embeddings)
np.random.seed(42)
#all_embeddings = np.random.rand(100, 50)

# Normalize the data (optional but often recommended for k-means)
scaler = StandardScaler()
all_embeddings_normalized = scaler.fit_transform(all_embeddings)

# Apply PCA for visualization (optional but useful for high-dimensional data)
# pca = PCA()
# embeddings_2d = pca.fit_transform(all_embeddings_normalized)


# Apply PCA
pca = PCA(n_components=.7)#70 pc variance
all_embeddings_pca = pca.fit_transform(all_embeddings_normalized)

# Calculate the cumulative explained variance
#cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components that explain more than 90% variance
#num_components_to_keep = np.argmax(cumulative_variance_ratio >= 0.70) + 1

# Select the top components
#selected_components = all_embeddings_pca[:, :num_components_to_keep]


# Determine the optimal number of clusters (k) - you can use various methods for this
# For simplicity, let's assume you know the number of clusters, say k=3


# Determine the optimal number of clusters using the elbow method
# Experiment with different values of k
k_values = range(1, 25)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(all_embeddings_normalized)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

#6 seems to be optimal

k = 6

# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(all_embeddings_normalized)

filtered_df['kmeans']=cluster_labels
filtered_df['kmeans'].value_counts()

# Visualize the clusters in 2D (for illustrative purposes with PCA)
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-Means Clustering of Document Embeddings')
#plt.legend()
plt.show()



#sns plot with clusters
import seaborn as sns

sns.set(context="paper", style="white")
sns.set(rc = {'figure.figsize' : (12, 12)})

ax = sns.scatterplot(data=fdf,
                x="abstract_vector_x",
                y="abstract_vector_y",
                hue="kmeans")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))




# =============================================================================
# lets try and do lda 
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load a sample dataset for illustration purposes

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df.abstract)

# Specify the number of topics (adjust as needed)
num_topics = 6

# Create an LDA model
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_matrix = lda_model.fit_transform(tfidf_matrix)

# Display the top words for each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[:-10 - 1:-1]  # Display top 10 words for each topic
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

# Topic #1: data, based, method, paper, problem, learning, model, performance, proposed, methods
# Topic #2: mathbb, mathcal, prove, let, groups, group, algebra, algebras, frac, leq
# Topic #3: quantum, spin, field, model, energy, magnetic, theory, phase, states, state
# Topic #4: smart, malware, blockchain, cities, jordan, wsd, mobility, submodular, wf, cpr
# Topic #5: galaxies, star, ray, mass, stars, emission, stellar, observations, galaxy, formation
# Topic #6: withdrawn, stat, syst, speaker, mb, author, trading, prize, conception, hypergeometric

# Get the most probable topic for each document
predicted_topics = lda_matrix.argmax(axis=1)

# Add the predicted topics to the original documents

filtered_df['lda']=predicted_topics
filtered_df['lda'].value_counts()

#sns
import seaborn as sns

sns.set(context="paper", style="white")
sns.set(rc = {'figure.figsize' : (12, 12)})

ax = sns.scatterplot(data=filtered_df,
                x="abstract_vector_x",
                y="abstract_vector_y",
                hue="lda")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# =============================================================================
# lets try lsa
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=.8, min_df=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df.abstract)

# Apply LSA (Truncated SVD)
num_topics = 6
lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
lsa_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

# Get the most important words for each topic
terms = tfidf_vectorizer.get_feature_names_out()

# Display the top terms for each topic
topic_terms = []
for i, topic in enumerate(lsa_model.components_):
    top_terms_idx = topic.argsort()[-5:][::-1]  # Display top 5 terms for each topic
    top_terms = [terms[idx] for idx in top_terms_idx]
    topic_terms.append(f"Topic {i + 1}: {' '.join(top_terms)}")

# Display the results
#result_df = pd.DataFrame({'Document': documents, 'LSA_Topic': lsa_topic_matrix.argmax(axis=1)})

# Add the predicted topics to the original documents

filtered_df['lsa']=lsa_topic_matrix.argmax(axis=1)
filtered_df['lsa'].value_counts()

print("\nTop Terms for Each Topic:")
for topic_term in topic_terms:
    print(topic_term)


import seaborn as sns

sns.set(context="paper", style="white")
sns.set(rc = {'figure.figsize' : (12, 12)})

ax = sns.scatterplot(data=filtered_df,
                x="abstract_vector_x",
                y="abstract_vector_y",
                hue="lsa")

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# =============================================================================
# lets try and build a classification model using diff techniques
# =============================================================================







