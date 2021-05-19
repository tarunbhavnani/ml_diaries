# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:07:50 2021

@author: ELECTROBOT
"""

import tensorflow as tf

from sklearn.model_selection import train_test_split
import pandas as pd
dat= pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\datasets\imdb\IMDB Dataset.csv")

def clean(doc):
    doc= txt=re.sub(r'[^a-z\., ]', " ", doc.lower())
    doc= txt=re.sub(r'\s+', " ", doc.lower())
    return doc
documents=[clean(i) for i in dat['review']]
targets=dat['sentiment'].values


# =============================================================================
# train test
# =============================================================================
train_df, test_df,train_labels, test_labels= train_test_split(documents, targets, test_size=.3, stratify=targets, random_state=1)

# =============================================================================
# tokenizer
# =============================================================================
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tok= Tokenizer(num_words= 3000, split= " ", oov_token= '-oov-')
tok.fit_on_texts(train_df)


# =============================================================================
# encoder
# =============================================================================
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
encoder.fit_transform(targets)

# =============================================================================
# data generator
# =============================================================================
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,  documents, targets,tokenizer, maxlen,encoder, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.documents = documents
        self.indices = [i for i in range(0, len(self.documents))]
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        self.targets = targets
        self.on_epoch_end()
        self.tok= tokenizer
        self.maxlen=maxlen
        self.encoder=encoder

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X =[self.documents[i] for i in batch]
        X= self.tok.texts_to_sequences(X)
        X= pad_sequences(X, maxlen=self.maxlen)
        
        y=[self.targets[i] for i in batch]
        y=self.encoder.transform(y)
        #y= pd.get_dummies(y)
        y=tf.keras.utils.to_categorical(y, num_classes=len(self.encoder.classes_))
        
        
        # for i, id in enumerate(batch):
        #     X[i,] = # logic
        #     y[i] = # labels

        return X, y



train_gen=DataGenerator(documents=train_df, targets= train_labels,tokenizer=tok, maxlen=1000,encoder=encoder, batch_size=128, num_classes=None, shuffle=True)
val_gen=DataGenerator(documents=test_df, targets= test_labels,tokenizer=tok, maxlen=1000,encoder=encoder, batch_size=128, num_classes=None, shuffle=True)


# =============================================================================
#  Model
# =============================================================================


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout, LSTM, Input, Conv1D,MaxPooling1D



model= Sequential()
model.add(Embedding(input_dim=3000, output_dim= 256, input_length= 1000))
model.add(Dropout(.2))
model.add(Conv1D(filters=128,kernel_size=5, activation='relu'  ))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128,kernel_size=5, activation='relu'  ))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer= 'adam')




# =============================================================================
# train
# =============================================================================



model.fit_generator(generator=train_gen,
                    validation_data=val_gen,
                    epochs=5)
                    



# =============================================================================
# embeddings
# =============================================================================


embeddings_index = dict()
f = open(r'C:\Users\ELECTROBOT\PycharmProjects\bot\tarun_nlp\glove.6B.50d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


tok.word_index
# create a weight matrix for words in training docs
vocab_size= len(tok.word_index)+1
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in tok.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector




model= Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim= 50,weights=[embedding_matrix], input_length= 1000))
model.add(Dropout(.2))
model.add(Conv1D(filters=128,kernel_size=5, activation='relu'  ))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128,kernel_size=5, activation='relu'  ))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer= 'adam')


# =============================================================================
#  multi CNN model
# =============================================================================




inp= Input(shape=(1000,1))
Conv1d= Conv1D(32,3,activation='relu', padding='same')(inp)
maxp=[]
maxp.append(MaxPooling1D(2)(Conv1d))
Conv1d= Conv1D(32,4,activation='relu', padding='same')(inp)
maxp.append(MaxPooling1D(2)(Conv1d))
Conv1d= Conv1D(32,5,activation='relu', padding='same')(inp)
maxp.append(MaxPooling1D(2)(Conv1d))
z= Concatenate(axis=1)(maxp)
flat= Flatten()(z)
dense= Dense(128,activation='relu')(flat)
out= Dense(2,activation='softmax')(dense)
model=Model(inp, out)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer= 'adam')
model.summary()



dat= np.empty([1000,100], dtype='float').reshape(100,-1,1)  #(100,1000,1)
label= np.random.choice([0,1],100)
label= tf.keras.utils.to_categorical(label)#(100,2)

model.fit(dat, label, batch_size=32)





# =============================================================================
# attention
#https://keras.io/api/layers/attention_layers/attention/
#Dot-product attention layer, a.k.a. Luong-style attention.


# =============================================================================

from tensorflow.keras.models import Model
import tensorflow as tf

# Variable-length int sequences.
query_input = tf.keras.Input(shape=(None,), dtype='int32')
value_input = tf.keras.Input(shape=(None,), dtype='int32')

# Embedding lookup.
#token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
token_embedding = tf.keras.layers.Embedding(vocab_size,1000)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(value_input)

# CNN layer.
cnn_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)

# Concatenate query and document encodings to produce a DNN input layer.
input_layer = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])

outp = tf.keras.layers.Dense(len(set(targets)), activation='softmax')(input_layer)


model = Model(inputs=[query_input,value_input], outputs=outp)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit_generator(generator=train_gen,
                    validation_data=val_gen,
                    epochs=10)
                    


