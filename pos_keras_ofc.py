#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:00:14 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
import nltk
 
tagged_sentences = nltk.corpus.treebank.tagged_sents()
 
print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))


sentences, sentence_tags=[],[]

for i in tagged_sentences:
  #print(i)
  sen=[]
  tag=[]
  for j in i:
    sen.append(j[0])
    tag.append(j[1])
  sentences.append(sen)
  sentence_tags.append(tag)

print(sentences[5])
print(sentence_tags[5])

#split
from sklearn.model_selection import train_test_split

train_sentences, test_sentences, train_tags, test_tags= train_test_split(sentences,sentence_tags, test_size=.2)


words, tags = set([]), set([])


for i in sentences:
  for j in i:
    words.add(j.lower())

for i in sentence_tags:
  for j in i:
    tags.add(j)


word2index= {w:i+2 for i,w in enumerate(words)}
word2index["-PAD-"]=0
word2index["-OOV-"]=1


tag2index={w:i+1 for i,w in enumerate(tags)}
tag2index["-PAD-"]=0


#now using these convert sentences and sentences tags to number format

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for i in train_sentences:
  #print(i)
  seq=[]
  for j in i:
    seq.append(word2index[j.lower()])
  train_sentences_X.append(seq)

for i in test_sentences:
  #print(i)
  seq=[]
  for j in i:
    seq.append(word2index[j.lower()])
  test_sentences_X.append(seq)

for i in train_tags:
  #print(i)
  seq=[]
  for j in i:
    seq.append(tag2index[j])
  train_tags_y.append(seq)

for i in test_tags:
  #print(i)
  seq=[]
  for j in i:
    seq.append(tag2index[j])
  test_tags_y.append(seq)




#cool
print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])




#pad sequences

MAX_LENGTH=[len(i) for i in sentences]
MAX_LENGTH=max(MAX_LENGTH)
#MAX_LENGTH=len(max(train_sentences_X, key=len))


from keras.preprocessing.sequence import pad_sequences

train_sentences_X= pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding="post")
test_sentences_X= pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding="post")
train_tags_y= pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding="post")
test_tags_y= pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding="post")

from keras.models import Sequential
model= Sequential()
from keras.layers import InputLayer
model.add(InputLayer(input_shape=(MAX_LENGTH,)))
from keras.layers import Embedding
model.add(Embedding(input_dim=len(word2index), output_dim=128))
from keras.layers import LSTM, Bidirectional
model.add(Bidirectional(LSTM(units=256,return_sequences=True)))
from keras.layers import TimeDistributed, Dense
model.add(TimeDistributed(Dense(units=len(tag2index), activation='softmax')))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""
import pandas as pd
YY= pd.get_dummies(train_tags_y).values()
this wont work here earlier data was one d now its train_tags.y.shape(3131,271)
"""


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

YY= to_categorical(test_tags_y, len(tag2index))



model.fit(test_sentences_X, YY,batch_size=32, epochs=10, validation_split=.2, verbose=3 )

#testing
test_sample="what are you doing here at this time of the day"

#first convert it to vectors and pad

tt=[]
s=[]
for i in test_sample.split():
  #print(i)
  if i in words:
    print(i)
    s.append(word2index[i.lower()])
  else:
    #print(i)
    s.append(word2index["-OOV-"])
tt.append(s)  




tt= pad_sequences(tt, maxlen=MAX_LENGTH, padding="post")


#predict

pred=model.predict(tt)
#not readable so we will convert
#convert to readable format
index={j:i for i,j in tag2index.items()}
import numpy as np
pred_r=[]
for i in pred:
  for j in i:
    #print(index[np.argmax(j)])
    pred_r.append(index[np.argmax(j)])
pred_r


#function for the same:


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
      #print(categorical_sequence)
        token_sequence = []
        for categorical in categorical_sequence:
          print(categorical)
          token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences
 
print(logits_to_tokens(sequences=pred, index={i: t for t, i in tag2index.items()}))



#all bakwas accuracy
#its because most of them are pads or zeros
#we will have to create another metrics which will see the accuracies without the pad prediction

"#very similarly we can give a statement and statement plus one word and thus predict next words"   






