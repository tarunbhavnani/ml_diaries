# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:07:19 2023

@author: tarun

"""

import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np

train=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\train_rule.csv")
test= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_rule.csv")


test_trigger={}
for i in range(len(test)):
    if test.iloc[i].trigger_state not in test_trigger:
        test_trigger[test.iloc[i].trigger_state]= test.iloc[i].trigger_state_id
for i in range(len(train)):
    if train.iloc[i].trigger_state not in test_trigger:
        test_trigger[train.iloc[i].trigger_state]= train.iloc[i].trigger_state_id






test_action={}
for i in range(len(test)):
    if test.iloc[i].action not in test_action:
        test_action[str(test.iloc[i].action)]= str(test.iloc[i].action_id)
for i in range(len(train)):
    if train.iloc[i].action not in test_action:
        test_action[str(train.iloc[i].action)]= str(train.iloc[i].action_id)





# =============================================================================

train["trig"]= [str(i)+"_"+str(j) for i,j in zip(train.trigger_device, train.trigger_state)]
train["act"]= [str(i)+"_"+str(j) for i,j in zip(train.action, train.action_device)]

train['rule']= [i+j for i,j in zip(train.trig, train.act)]


kl= train.groupby('user_id')['rule'].apply(lambda x: list(x)).reset_index()

kl=kl[[True if len(i)>1 else False for i in kl.rule]]

x=[]
y=[]
for i in range(len(kl)):
    temp=kl.rule.iloc[i]
    for num,j in enumerate(temp):
        lst=temp.copy()
        yy=lst.pop(num)
        x.append(lst)
        y.append(yy)
        




#data= [i for i in kl.rule]

 

documents=[",".join(i) for i in x]

# vocab= {}
# counter=0
# for io in data:
#     for i in io:
#         if i not in vocab:
#             vocab[i]=counter
#             counter+=1
            

# data_= [[vocab[j] for j in i] for i in data]




import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB





# # Example sentences (replace with your own data)
# sentences = [
#     "The quick brown fox jumps over the lazy dog.",
#     "She sells seashells by the seashore.",
#     "How much wood would a woodchuck chuck if a woodchuck could chuck wood."
# ]

# # Create a list of target words corresponding to the missing words
# target_words = ["dog", "sells", "woodchuck"]

# Tokenize the sentences using a CountVectorizer

def custom_tokenizer(text):
    # Split text by spaces and return the tokens
    return text.split(",")
vectorizer = CountVectorizer(tokenizer=custom_tokenizer,ngram_range=(1, 3))

#vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)



# Initialize the classifier
classifier = MultinomialNB()

# Define the batch size (you can adjust this based on your available memory)
batch_size = 100000

# Split the data into batches and train the classifier incrementally
for i in range(0, len(y), batch_size):
    print(".", end="", flush=True)
    batch_X = X[i:i+batch_size]
    batch_target_words = y[i:i+batch_size]
    classifier.partial_fit(batch_X, batch_target_words, classes=np.unique(y))


# Predict the missing words in a new sentence
new_sentence = documents[2]

# Tokenize the new sentence using the same vectorizer
new_X = vectorizer.transform([new_sentence])

# Predict missing words
predicted_probabilities = classifier.predict_proba(new_X)
import numpy as np
print(f"The missing words are: {predicted_words}")
# Get the indices of the top ten predictions
top_n_indices = np.argsort(predicted_probabilities[0])[-10:]

# Get the corresponding target words
top_n_words = [classifier.classes_[i] for i in top_n_indices]




# =============================================================================
# ideally break above to test and train, get accuracy of test data and the proceed 
# =============================================================================




# =============================================================================
#lets predict on test data 
# =============================================================================


test["trig"]= [str(i)+"_"+str(j) for i,j in zip(test.trigger_device, test.trigger_state)]
test["act"]= [str(i)+"_"+str(j) for i,j in zip(test.action, test.action_device)]

test['rule']= [i+j for i,j in zip(test.trig, test.act)]



nd= test.groupby('user_id')['rule'].apply(lambda x: list(x)).reset_index()
nd=[",".join(i) for i in nd.rule]



# Transform the batch of sentences
new_X = vectorizer.transform(nd)

# Predict missing words for the batch
predicted_probabilities = classifier.predict_proba(new_X)

top_n_indices_batch = np.argsort(predicted_probabilities, axis=1)[:, -100:]
# Get the corresponding target words for each sentence in the batch
top_n_words_batch = [
    [classifier.classes_[i] for i in top_n_indices] for top_n_indices in top_n_indices_batch
]



top_n_words_batch[0]


#now convert these to rules and submit

