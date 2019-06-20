#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:39:26 2019

@author: tarun.bhavnani
"""

import unicodedata
import re

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    #w = '<start> ' + w + ' <end>'
    return w

sent=[preprocess_sentence(i) for i in sentences]
sent= [re.sub("[^a-z\s]","",x) for x in sent]

#sent=[re.sub(r"\s.",".",preprocess_sentence(i)) for i in sent]



###############################################################################
###############################################################################

#Create a word2Vec 
#https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
#split again for word2vec

sent_split=[i.split() for i in sent]


from gensim.models import Word2Vec
import multiprocessing

w2v= Word2Vec(sent_split, size=300, window=5,
              min_count=5, negative=15, iter=10, workers= multiprocessing.cpu_count())
print(w2v)

words = list(w2v.wv.vocab)
print(w2v['sentence'])

#create a word embedding for some words
wd= words[1:100]

dat=[w2v[i] for i in wd]
df=pd.DataFrame(data=dat,index=wd)
#df is the req word embedding of words-wd

#also
result= w2v.wv.similar_by_word("saturday")


###############################################################################
###############################################################################


#using pre trained embeddings--Glove


embeddings_index = {}
glove_dir="/home/tarun.bhavnani/Desktop/destop/glove_data"
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'), encoding="Latin1")
for line in f:
    #print(line)
    values = line.split()
    word = values[0]
    try:
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs
    except:
        print(word)
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM=100
embedding_matrix = np.zeros((len(words), EMBEDDING_DIM))

for i,word in enumerate(words):
    print(i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


df= pd.DataFrame(data= embedding_matrix, index= words)




###############################################################################
###############################################################################


#using pre trained embeddings in keras models

#Read the news data

############################33
import sys
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
#labels = []  # list of label ids
TEXT_DATA_DIR = "C:\\Users\\tarun.bhavnani\\Desktop\\keras\\news20\\20_newsgroup"
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    print(path)
    if os.path.isdir(path):
        print("True")
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))


########################################################
#format sentences so they can be fed
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
MAX_NB_WORDS=2000
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH=2000
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
VALIDATION_SPLIT=.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


embeddings_index = {}
glove_dir="C:\\Users\\tarun.bhavnani\\Desktop\\embed_kera\\glove.6B"
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'), encoding="Latin1")
for line in f:
    #print(line)
    values = line.split()
    word = values[0]
    try:
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs
    except:
        print(word)
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM=100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    print(i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix.shape

#embedding layer has input dim as the word index length, output dim as the latent dim
#if we create aan embedidng outside or use a pre created embedding matriz we can
#just input he whole thing as weights in the embedding layer and trainable =False to use
#as is


from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x1, state_h, state_C=LSTM(256, return_state=True)(embedded_sequences)


from keras.layers import Conv1D, MaxPooling1D, Flatten
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128)


















