#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:50:22 2019

@author: tarun.bhavnani@dev.smecorner.com

#https://developers.google.com/machine-learning/guides/text-classification/step-4
"""
#CNN on sparse matrix, i.e tfidf matrix instead of word vec embeddings
#Prepare data



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
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, analyzer="word",
                             strip_accents="unicode", decode_error="replace")

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

model=sepcnn_model(blocks=2,
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
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
model.summary()


history = model.fit(
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

Thats why we will use embeddings
se Transc_CNN_model3.py in the same repo.


"""






    