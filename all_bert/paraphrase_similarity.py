# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:39:40 2021

@author: ELECTROBOT
"""

import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Single list of sentences
sentences = ['The umbrella sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat sits outside',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

#Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:10]:
    i, j = pair['index']
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
    
# =============================================================================
# 
# =============================================================================
question= "whare does the cat sit"
question_embedding= model.encode([question], convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(question_embedding, embeddings)
import numpy as np
kl=torch.argmax(cosine_scores).detach().cpu().numpy()
sentences[kl.reshape(-1)[0]]



# 
question= "whare does the cat play"
question_embedding= model.encode([question], convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(question_embedding, embeddings)
import numpy as np
kl=torch.argmax(cosine_scores).detach().cpu().numpy()
sentences[kl.reshape(-1)[0]]




