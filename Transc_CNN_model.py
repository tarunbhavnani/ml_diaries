#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:50:22 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):
    """Creates an instance of a separable CNN model.

    # Arguments
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of the layers.
        kernel_size: int, length of the convolution window.
        embedding_dim: int, dimension of the embedding vectors.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.
        num_features: int, number of words (embedding input dimension).
        use_pretrained_embedding: bool, true if pre-trained embedding is on.
        is_embedding_trainable: bool, true if embedding layer is trainable.
        embedding_matrix: dict, dictionary with embedding coefficients.

    # Returns
        A sepCNN model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = Sequential()

    # Add embedding layer. If pre-trained embedding is used add weights to the
    # embeddings layer and set trainable to input is_embedding_trainable flag.
    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[1],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable))
    else:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[1]))

    for _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model


#Prepare data


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, MaxPooling1D, GlobalAveragePooling1D,SeparableConv1D

tok= Tokenizer(num_words=2000, split=" ", oov_token="-OOV-")
tok.fit_on_texts(fdf.Des.values)
X=tok.texts_to_sequences(fdf.Des.values)
X=pad_sequences(X, maxlen=10)

Y= pd.get_dummies(fdf.classification.values)




model=sepcnn_model(blocks=2,
                 filters=64,
                 kernel_size=5,
                 embedding_dim=64,
                 dropout_rate=.2,
                 pool_size=2,
                 input_shape=X.shape,
                 num_classes=Y.shape[1],
                 num_features=2000,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None)


#optimizer = tf.keras.optimizers.Adam(lr=.001)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
from keras import callbacks
callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
history = model.fit(
            X,
            Y,
            epochs=10,
            callbacks=callbacks,
            validation_split=.33,
            verbose=1,  # Logs once per epoch.
            batch_size=32)

# Print results.
hist = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=hist['val_acc'][-1], loss=hist['val_loss'][-1]))

model.save('Transc_mlp_model.h5')
#filters:32 kernel 3
#Validation accuracy: 0.9873930350836291, loss: 0.07071986348755928

model.save('Transc_mlp_model64_3.h5')
#filters:64 kernel 3
#Validation accuracy: 0.9885587475930195, loss: 0.0702287603003148
#filters:64 kernel 5

model.save('Transc_mlp_model62_5_256.h5')
#Validation accuracy: 0.9890250325967758, loss: 0.0683740159372448
#filters:62 kernel 5, embed_dim:256

# Save model.
model.save('Transc_mlp_model.h5')


#############Plotting#############3
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#predict

test= fdf.Des[50:51].values
tt=tok.texts_to_sequences(test)
#inde={j:i for i,j in tok.word_index.items()}

#tt=[]

tt=pad_sequences(tt, maxlen=10)

pred= model.predict(tt)

index={j:i for i,j in tok.word_index.items()}
import numpy as np
pred_r=[]
for i in pred:
  #print(i)
  for j in i:
    #print(j)
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
 
print(logits_to_tokens(sequences=pred, index={j:i for i,j in tok.word_index.items()}))









#lets d o a same thing with tfidf word vectors


from sklearn.feature_extraction.text import TfidfVectorizer

vec= TfidfVectorizer(min_df=500)
X1= vec.fit_transform(fdf.Des.values)
print(vec.get_feature_names())

X1.shape
X.shape

model1=sepcnn_model(blocks=2,
                 filters=32,
                 kernel_size=3,
                 embedding_dim=128,
                 dropout_rate=.2,
                 pool_size=2,
                 input_shape=X1.shape,
                 num_classes=Y.shape[1],
                 num_features=2000,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None)


#optimizer = tf.keras.optimizers.Adam(lr=.001)
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
model1.summary()


history1 = model1.fit(
            X1,
            Y,
            epochs=4,
            callbacks=callbacks,
            validation_split=.33,
            verbose=1,  # Logs once per epoch.
            batch_size=32)

#####################################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    """
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, analyzer="word", strip_accents="unicode", decode_error="replace")

    # Learn vocabulary from training texts and vectorize training texts.
x_train = vectorizer.fit_transform(fdf.Des.values)

    # Vectorize validation texts.
    #x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
selector = SelectKBest(f_classif, k=min(500, x_train.shape[1]))
train_labels= fdf.classification.values
selector.fit(x_train, train_labels)
x_train = selector.transform(x_train).astype('float32')
    #x_val = selector.transform(x_val).astype('float32')
    #return x_train, x_val

model1=sepcnn_model(blocks=2,
                 filters=32,
                 kernel_size=3,
                 embedding_dim=128,
                 dropout_rate=.2,
                 pool_size=2,
                 input_shape=x_train.shape,
                 num_classes=Y.shape[1],
                 num_features=2000,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None)


#optimizer = tf.keras.optimizers.Adam(lr=.001)
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
model1.summary()


history1 = model1.fit(
            x_train,
            Y,
            epochs=4,
            callbacks=callbacks,
            validation_split=.33,
            verbose=1,  # Logs once per epoch.
            batch_size=32)
#very bad accuracy, I think CNN on a sparse matrix can't give goof results.

"""
Train on 235126 samples, validate on 115809 samples
Epoch 1/4
235126/235126 [==============================] - 538s 2ms/step - loss: 1.9949 - acc: 0.3630 - val_loss: 1.9417 - val_acc: 0.4220
Epoch 2/4
235126/235126 [==============================] - 519s 2ms/step - loss: 1.9694 - acc: 0.3659 - val_loss: 1.9425 - val_acc: 0.4220....] - ETA: 6:06 - loss: 1.9663 - acc: 0.3685

"""






    