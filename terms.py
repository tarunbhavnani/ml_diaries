#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:35:53 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
rasa core and nlu deep learning models, how they train on the data
ner-everything:classify named entities in text into pre-defined categories such as the names of persons, 
organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

can use nltk
nltk.word_tokenize, pos_tag, etc
oe spacy:
  nlp=en_core_web_sm.load()





relation extraction-- same as above

text summary
text classification


lda-- is a generative probabilistic model, that assumes a Dirichlet prior over the latent topics..lda finds the latent topics in the documents. it tries to make a words*w, w*docs such a way 
to minimize the error. we have different topics which are clusters of diff words and each document 
has a propb of fallinf into each one of topics.

lsa- its more like PCA.learns latent topics by performing a matrix decomposition (SVD) 
on the term-document matrix.

lsa is faster but lesser accuracy mostly.

pca:Principal components analysis is a procedure for identifying a smaller number of uncorrelated variables, called “principal components”, from a large set of data. The goal of principal components analysis is to explain the maximum amount of variance with the fewest number of principal components.

imbalanced data--propensity to buy at tcs, and rest of tcs projects

svm: Separation of classes. That’s what SVM does. Kernels(linera, polynomial) Polynomial and exponential kernels calculates separation line in higher dimension. This is called kernel trick
gamma: low gamma considers points close to plane, high gamma considers farther points
C: big c low margin, low C high margin

svc
cnn
lstm
bilstm
nn
rnn
optimizers
batch normalization
gd/sgd/adam/rmsprop/adagrad

randomforest
dt
xgboost

seq2seq, return sequences, return state
nmt
autoencoders

dropout

confusion matrix

lift--The basic idea of lift analysis is as follows:

group data based on the predicted churn probability (value between 0.0 and 1.0). Typically, you look at deciles, so you'd have 10 groups: 0.0 - 0.1, 0.1 - 0.2, ..., 0.9 - 1.0
calculate the true churn rate per group. That is, you count how many people in each group churned and divide this by the total number of customers per group.







Rasa NLU
The two components between which you can choose are:

Pretrained Embeddings (Intent_classifier_sklearn)
Supervised Embeddings (Intent_classifier_tensorflow_embedding)

Word embeddings are vector representations of words, meaning each word is converted to a dense numeric vector. Word embeddings capture semantic and syntactic aspects of words. This means that similar words should be represented by similar vectors.

intent_classifier_tensorflow_embedding--t trains word embeddings from scratch. It is typically used with the intent_featurizer_count_vectors component which counts how often distinct words of your training data appear in a message and provides that as input for the intent classifier.


Extracting Entities
ner_spacy: pre-traines. Entity recognition with SpaCy language models

ner_http_duckling: Rule based entity recognition using Facebook’s Duckling: -->amounts of money, dates, distances, or durations

ner_crf: raining an extractor for custom entities: --> Neither ner_spacy nor ner_duckling require you to annotate any of your training data, since they are either using pretrained classifiers (spaCy) or rule-based approaches (Duckling). The ner_crf component trains a conditional random field which is then used to tag entities in the user messages. Since this component is trained from scratch as part of the NLU pipeline you have to annotate your training data yourself. This is an example from our documentation on how to do so
also has regex support and lookup tables

ner_synonyms


generative and discriminative: https://en.wikipedia.org/wiki/Discriminative_model
The typical discriminative learning approaches include Logistic Regression (LR), Support Vector Machine (SVM), conditional random fields (CRFs) (specified over an undirected graph), and others. The typical generative model approaches contain Naive Bayes, Gaussian Mixture Model, and others.



#sklearn  feature on text
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.category_id
features.shape

#correlated terms with each category and more... full code on:
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f



precision: TP/TP+FP
recall:    TP/TP+FN
f1: harmonic mean of precision and recall


Imbalanced data: One of the tactics of combating imbalanced classes is using Decision Tree algorithms, so, we are using Random Forest classifier to learn imbalanced data and set class_weight=balanced 


pca--> 400 images 64*64---> 400 vectors of 4096 elements--> 400 pca
Each of the 400 original images (i.e. each of the 400 original rows of the matrix) can be expressed as a (linear) combination of the 400 pca's.
The goal of PCA is to reveal typical vectors: each of the creepy/typical guy represents one specific aspect underlying the data.
each of the PCA's captures a specific aspect of the data. Each principal component captures a specific latent factor.


Matrix--> users on rows and movies on cols
PCA on R=M
PCA on R(T)---> PCA on Matrix--> movies on rows and users on cols--> U(T)

SVD on this R matrix will give--> MU(T) or MEU(T) where E is a diagonal matrix
basically what it does is:

rui  =  pu⋅qi  =  ∑f∈latent factorsaffinity of u for f×affinity of i for f
affinity of user for f into affinity of movie for f




#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

Neural Network
A neural network usually involves a large number of processors operating in parallel and arranged in tiers. The first tier receives the raw input information -- analogous to optic nerves in human visual processing. Each successive tier receives the output from the tier preceding it, rather than from the raw input -- in the same way neurons further from the optic nerve receive signals from those closer to it. The last tier produces the output of the system.


RNN


LSTM
forget gate
The portion of a Long Short-Term Memory cell that regulates the flow of information through the cell. Forget gates maintain context by deciding which information to discard from the cell state.

CNN


Autoencoder, encoder/decoder


generative adversarial network (GAN)
A system to create new data in which a generator creates data and a discriminator determines whether that created data is valid or invalid.

Attention

Word Embedding


Word2vec

backpropagation
The primary algorithm for performing gradient descent on neural networks. First, the output values of each node are calculated (and cached) in a forward pass. Then, the partial derivative of the error with respect to each parameter is calculated in a backward pass through the graph.
Seq2Seq

activation function

A function (for example, ReLU or sigmoid) that takes in the weighted sum of all of the inputs from the previous layer and then generates and passes an output value (typically nonlinear) to the next layer.

Optimizer


Adam
rmsprop

Gradient Descent: A technique to minimize loss by computing the gradients of loss with respect to the model's parameters, conditioned on training data. Informally, gradient descent iteratively adjusts parameters, gradually finding the best combination of weights and bias to minimize loss.
Gradient:the gradient is the vector of partial derivatives of the model function. The gradient points in the direction of steepest ascent.

SGD
AdaGrad
A sophisticated gradient descent algorithm that rescales the gradients of each parameter, effectively giving each parameter an independent learning rate.




batch normalization
Normalizing the input or output of the activation functions in a hidden layer. Batch normalization can provide the following benefits:

Make neural networks more stable by protecting against outlier weights.
Enable higher learning rates.
Reduce overfitting.

Maxpooling
Flatten

filters/kernels

SVM
SVC

SVD

PCA

Random Forest

Logistic Regression

Linera Regression

Decision Tree

AUC (Area under the ROC Curve)
An evaluation metric that considers all possible classification thresholds.

The Area Under the ROC curve is the probability that a classifier will be more confident that a randomly chosen positive example is actually positive than that a randomly chosen negative example is positive.
Clustering
KNN

Kmeans


Parametric/Non Parametric


Supervised/Non Supervised

boosting
A ML technique that iteratively combines a set of simple and not very accurate classifiers (referred to as "weak" classifiers) into a classifier with high accuracy (a "strong" classifier) by upweighting the examples that the model is currently misclassfying.


collaborative filtering
Making predictions about the interests of one user based on the interests of many other users. Collaborative filtering is often used in recommendation systems.


confirmation bias
#fairness
The tendency to search for, interpret, favor, and recall information in a way that confirms one's preexisting beliefs or hypotheses. Machine learning developers may inadvertently collect or label data in ways that influence an outcome supporting their existing beliefs. Confirmation bias is a form of implicit bias.

Experimenter's bias is a form of confirmation bias in which an experimenter continues training models until a preexisting hypothesis is confirmed.

confusion matrix
An NxN table that summarizes how successful a classification model's predictions were;


convex function
A function in which the region above the graph of the function is a convex set. The prototypical convex function is shaped something like the letter U.
A strictly convex function has exactly one local minimum point, which is also the global minimum point. 
 lot of the common loss functions, including the following, are convex functions:

L2 loss
Log Loss
L1 regularization
L2 regularization
Many variations of gradient descent are guaranteed to find a point close to the minimum of a strictly convex function. 
Deep models are never convex functions. Remarkably, algorithms designed for convex optimization tend to find reasonably good solutions on deep networks anyway, even though those solutions are not guaranteed to be a global minimum.


convex optimization
The process of using mathematical techniques such as gradient descent to find the minimum of a convex function.


convolutions
In machine learning, a convolution mixes the convolutional filter and the input matrix in order to train weights.For example, a machine learning algorithm training on 2K x 2K images would be forced to find 4M separate weights. Thanks to convolutions, a machine learning algorithm only has to find weights for every cell in the convolutional filter, dramatically reducing the memory needed to train the model. When the convolutional filter is applied, it is simply replicated across cells such that each is multiplied by the filter.

convolutional filter
One of the two actors in a convolutional operation. (The other actor is a slice of an input matrix.) A convolutional filter is a matrix having the same rank as the input matrix, but a smaller shape. For example, given a 28x28 input matrix, the filter could be any 2D matrix smaller than 28x28.


cross-entropy
A generalization of Log Loss to multi-class classification problems. Cross-entropy quantifies the difference between two probability distributions.

dense layer
Synonym for fully connected layer.
fully connected layer
A hidden layer in which each node is connected to every node in the subsequent hidden layer.

depth
The number of layers (including any embedding layers) in a neural network that learn weights. For example, a neural network with 5 hidden layers and 1 output layer has a depth of 6.


depthwise separable convolutional neural network (sepCNN)
A convolutional neural network architecture based on Inception, but where Inception modules are replaced with depthwise separable convolutions. Also known as Xception.

A depthwise separable convolution (also abbreviated as separable convolution) factors a standard 3-D convolution into two separate convolution operations that are more computationally efficient: first, a depthwise convolution, with a depth of 1 (n ✕ n ✕ 1), and then second, a pointwise convolution, with length and width of 1 (1 ✕ 1 ✕ n).

To learn more, see Xception: Deep Learning with Depthwise Separable Convolutions.

dropout regularization
A form of regularization useful in training neural networks. Dropout regularization works by removing a random selection of a fixed number of the units in a network layer for a single gradient step. 


feature vector
The list of feature values representing an example passed into a model.


generalization curve
A loss curve showing both the training set and the validation set. A generalization curve can help you detect possible overfitting. 



heuristic
A quick solution to a problem, which may or may not be the best solution.


hidden layer
A synthetic layer in a neural network between the input layer (that is, the features) and the output layer (the prediction). Hidden layers typically contain an activation function (such as ReLU) for training. A deep neural network contains more than one hidden layer.




L1 loss
Loss function based on the absolute value of the difference between the values that a model is predicting and the actual values of the labels. L1 loss is less sensitive to outliers than L2 loss.

L1 regularization
A type of regularization that penalizes weights in proportion to the sum of the absolute values of the weights. In models relying on sparse features, L1 regularization helps drive the weights of irrelevant or barely relevant features to exactly 0, which removes those features from the model. Contrast with L2 regularization.

L2 loss
See squared loss.


L2 regularization
A type of regularization that penalizes weights in proportion to the sum of the squares of the weights. L2 regularization helps drive outlier weights (those with high positive or low negative values) closer to 0 but not quite to 0. (Contrast with L1 regularization.) L2 regularization always improves generalization in linear models.


layer
A set of neurons in a neural network that process a set of input features, or the output of those neurons.

Also, an abstraction in TensorFlow. Layers are Python functions that take Tensors and configuration options as input and produce other tensors as output. Once the necessary Tensors have been composed, the user can convert the result into an Estimator via a model function.

learning rate
A scalar used to train a model via gradient descent. During each iteration, the gradient descent algorithm multiplies the learning rate by the gradient. The resulting product is called the gradient step.



linear regression
A type of regression model that outputs a continuous value from a linear combination of input features.


Logistic regression
A model that generates a probability for each possible discrete label value in classification problems by applying a sigmoid function to a linear prediction. Although logistic regression is often used in binary classification problems, it can also be used in multi-class classification problems (where it becomes called multi-class logistic regression or multinomial regression).

logits
The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.

In addition, logits sometimes refer to the element-wise inverse of the sigmoid function. For more information, see tf.nn.sigmoid_cross_entropy_with_logits.



log-odds
The logarithm of the odds of some event.

If the event refers to a binary probability, then odds refers to the ratio of the probability of success (p) to the probability of failure (1-p). For example, suppose that a given event has a 90% probability of success and a 10% probability of failure. In this case, odds is 9

The log-odds is simply the logarithm of the odds. By convention, "logarithm" refers to natural logarithm, but logarithm could actually be any base greater than 1. Sticking to convention, the log-odds of our example is therefore: ln(9)=2.2

The log-odds are the inverse of the sigmoid function.

Long Short-Term Memory (LSTM)
A type of cell in a recurrent neural network used to process sequences of data in applications such as handwriting recognition, machine translation, and image captioning. LSTMs address the vanishing gradient problem that occurs when training RNNs due to long data sequences by maintaining history in an internal memory state based on new input and context from previous cells in the RNN.



matrix factorization
in recommender system we have sparse users vs movies, we want it dense.
we find users*n matrix and n*movies matrix 
suxh that the dot product is as close as the sparse users*movies matrix.


Momentum
A sophisticated gradient descent algorithm in which a learning step depends not only on the derivative in the current step, but also on the derivatives of the step(s) that immediately preceded it. Momentum involves computing an exponentially weighted moving average of the gradients over time, analogous to momentum in physics. Momentum sometimes prevents learning from getting stuck in local minima.


Neural Network
A model that, taking inspiration from the brain, is composed of layers (at least one of which is hidden) consisting of simple connected units or neurons followed by nonlinearities.

normalization
The process of converting an actual range of values into a standard range of values, typically -1 to +1 or 0 to 1. For example, suppose the natural range of a certain feature is 800 to 6,000. Through subtraction and division, you can normalize those values into the range -1 to +1.

See also scaling.


objective function
The mathematical formula or metric that a model aims to optimize. For example, the objective function for linear regression is usually squared loss. Therefore, when training a linear regression model, the goal is to minimize squared loss.

In some cases, the goal is to maximize the objective function. For example, if the objective function is accuracy, the goal is to maximize accuracy.



optimizers




perceptron
A system (either hardware or software) that takes in one or more input values, runs a function on the weighted sum of the inputs, and computes a single output value. In machine learning, the function is typically nonlinear, such as ReLU, sigmoid, or tanh. For example, the following perceptron relies on the sigmoid function to process three input values:

f(x1,x2,x3)= sigmoid(w1x1+w2x2+w3x3)

pooling
Reducing a matrix (or matrices) created by an earlier convolutional layer to a smaller matrix. Pooling usually involves taking either the maximum or average value across the pooled area.
















