

#cosine

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define two vectors
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 3, 6])

# Calculate cosine similarity
similarity = cosine_similarity([vector1, vector2])


#manually

dot_product = np.dot(vector1, vector2)

# Calculate magnitudes
magnitude1 = np.sqrt(np.sum(np.square(vector1)))
magnitude2 = np.sqrt(np.sum(np.square(vector2)))

# Calculate cosine similarity
cosine_similarity = dot_product / (magnitude1 * magnitude2)

print(cosine_similarity)



#jaccard

#size of interaction/ size of union

from sklearn.metrics import jaccard_score
y_true = np.array([[0, 1, 1],
                    [1, 1, 0]])
y_pred = np.array([[1, 1, 1],
                   [1, 0, 0]])
#In the binary case:

jaccard_score(y_true[0], y_pred[0])


