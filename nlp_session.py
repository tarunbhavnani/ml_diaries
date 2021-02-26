#!/usr/bin/env python
# coding: utf-8

# ## SMEcorner NLP Session1- Text Classification
# 
# Objectives:
# 
#     - Machines understanding of text
#     
#     - Basic text cleaning
#     
#     - basic to advanced text classification

# In[144]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# ### Classification is a Supervised Predictive modelling exercise.
# ### To understand text classification lets first look at a ml classification example. 
# ### Here we will try and see how we will classify the flowers types in the famous IRIS dataset.

# In[145]:



iris = load_iris()
iris_df=pd.DataFrame(iris['data'],  columns= iris['feature_names'])
iris_df['labels']= iris['target']
#'setosa', 'versicolor', 'virginica'


# In[4]:


iris_df.sample(frac=1).head()


# In[21]:


#initiate logistic regreesion
clf = LogisticRegression(random_state=0)

#fit our parameters
clf.fit(iris_df.iloc[:, 0:4].values, iris_df.iloc[:, 4].values)


# In[22]:


#Pridict flower for sepal length (cm)=8, sepal width (cm)= 4, petal length (cm)=7, petal width (cm)=2

check_candidate=([[8,4,7,2]])


# In[23]:


clf.predict(check_candidate)[0] #'virginica'


# ### In the few lines of code above, we created a classification algorithm for the iris dataset.
# 
# ### How do you think the text classification differs from this?
# 
# 

# # lets first load and understand our data

# In[25]:


dat= pd.read_csv(r'C:\Users\ELECTROBOT\Desktop\nlp_session\news\train.csv')

dat.columns= ['labels', 'title', 'description']
dat.sample(frac=1).head(50)
#'World','Sports','Business','Sci/Tech'


# ### Not by alot.
# 
# ### The features vector above were petal, sepal length and width. 
# 
# ### In text classification we have to create a feature vector from our documents(sentences)
# 
# ### Words are our features.
# 
# ### Now we will see the different ways we convert words to features, also called text vectorization methods. 

# ### We will use the titles and train the model to predict the lables.

# # Objective: How does machine understand language
# 
# 
# We have to classifiy the sentences. Every word possible is a feature and the count of words in the sentence makes the feature vector.
# 

# In[58]:


#Frequency of words
words={}
for sent in dat.title.values:
    for word in sent.split():
        if word in words:
            words[word]+=1
        else:
            words[word]=1
# Creating a term document matrix, we will sue just the first 100 observations
final_array=[len([i for i in dat.iloc[0].title.split() if i==word]) for word in words]
for sent in dat[1:100].title.values:
    print('.', end="", flush=True)
    ary= [len([i for i in sent.split() if i==word]) for word in words]
    final_array= np.vstack((final_array, ary))

    
tdm=pd.DataFrame(final_array, columns= words)


# In[59]:


tdm.index= dat['title'].iloc[0:100]
tdm.head()


# ### Looks pretty similar to the isir case. Can we start using a log reg to train and predict----YES we can.
# 
# ### we can say that the machine is understanding language(to some extent)
# 
# ### Lets try and improve the understanding for this case!
# 
# ### 100 observations and 71744 features, it is alot? 
# ### we clean!!
# 

# ### Basics of text cleaning!!
# 
# - Case sensitive, lower the text
# - text has alphabets, digits, punctuations and otherwise special characters, what all to keep?
# - Extra spaces
# - To Stopwords or not?
# - To stem/lemma or not?

# In[50]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

def clean(sent):
    sent=sent.lower()
    sent=re.sub(r'[^a-z0-9 ]', " ",sent)
    sebt=re.sub(r'\s+', " ",sent)
    sent=" ".join([stemmer.stem(s) for s in sent.split()])
    
    return sent


# In[53]:


#dat['title_clean']= [clean(sent) for sent in dat.description]
dat['title_clean']= [clean(sent) for sent in dat.title]


# In[54]:


print(dat.title.iloc[0:5])
print(dat.title_clean.iloc[0:5])


# In[55]:


#Frequency of words
words={}
for sent in dat.title_clean.values:
    for word in sent.split():
        if word in words:
            words[word]+=1
        else:
            words[word]=1
len(words)


# ### Now we have a good understading of how things function in cleaning and machine understading of the feature vectors. 
# 
# 
# ### lets use sklearn proceed !!

# The Scikit-Learn's CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
# 
# 

# We take a dataset and convert it into a corpus. Then we create a vocabulary of all the unique words in the corpus. Using this vocabulary, we can then create a feature vector of the count of the words. Let's see this through a simple example. Let's say we have a corpus containing two sentences as follows

# In[80]:


from sklearn.feature_extraction.text import CountVectorizer

#fit

vectorizer = CountVectorizer()
vectorizer.fit(dat.title_clean.values)
#vectorizer.vocabulary_


# In[81]:


#transform

train_vectors = vectorizer.transform(dat.title_clean.values)

train_vectors.shape


# By default, a scikit learn Count vectorizer can perform the following opertions over a text corpus:
# 
# Encoding via utf-8
# 
# converts text to lowercase
# 
# Tokenizes text using word level tokenization
# 
# CountVectorizer has a number of parameters. Let's look at some of them :

# # Stopwords
# 
# we can remove stopwords depending on our requirements. Can use NLTK or a custom list as well

# In[70]:


#from nltk.corpus import stopwords
#print(stopwords.words('english'))
stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

#fit
count_vectorizer = CountVectorizer(stop_words = stopwords)
count_vectorizer.fit(dat.title_clean.values)


#transform
train_vectors = count_vectorizer.transform(dat.title_clean.values)


train_vectors.shape


# ### MIN_DF and MAX_DF parameter
# MIN_DF lets you ignore those terms that appear rarely in a corpus. In other words, if MIN_dfis 2, it means that a word has to occur at least two documents to be considered useful.
# 
# MAX_DF on the other hand, ignores terms that have a document frequency strictly higher than the given threshold.These will be words which appear a lot of documents.
# 
# This means we can eliminate those words that are either rare or appear too frequently in a corpus.
# 
# When mentioned in absolute values i.e 1,2, etc, the value means if the word appears in 1 or 2 documents. However, when given in float, eg 30%, it means it appears in 30% of the documents.

# In[71]:


#fit
count_vectorizer = CountVectorizer(stop_words = stopwords, min_df=2 ,max_df=0.8)
count_vectorizer.fit(dat.title_clean.values)


#transform
train_vectors = count_vectorizer.transform(dat.title_clean.values)


train_vectors.shape


# ### Custom preprocessor
# 
# Cleaning and other realted tasks

# In[82]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

def clean(sent):
    sent=sent.lower()
    sent=re.sub(r'[^A-Za-z ]', " ",sent)
    sent=re.sub(r'\s+', " ",sent)
    #sent=" ".join([stemmer.stem(s) for s in sent.split() if s not in stopwords])
    
    
    return sent


# In[124]:


#fit
count_vectorizer = CountVectorizer( min_df=20 ,max_df=0.8, preprocessor= clean)
count_vectorizer.fit(dat.title_clean.values)


#transform
train_vectors = count_vectorizer.transform(dat.title_clean.values)


train_vectors.shape


# In[125]:


#count_vectorizer.vocabulary_


# ### lets put our classification model to work

# In[126]:


y= dat.labels.values
y


# In[127]:


from sklearn.model_selection import train_test_split


Xtr, Xtt, Ytr, Ytt= train_test_split(train_vectors, y, test_size=.3, stratify= y, random_state=98)


# In[128]:


clf = LogisticRegression(random_state=0).fit(Xtr, Ytr)


# In[129]:


y_pred= clf.predict(Xtt)


# In[130]:


from sklearn.metrics import accuracy_score, f1_score

print(accuracy_score(Ytt, y_pred)*100)
print(f1_score(Ytt, y_pred, average='macro')*100)


# ### ---------We have built our first text classification model ---------------
# 
# Things we learnt here:
#     
#     The features are all the words(too many). The word vectors are the count of occurance of thme word(pretty naive).
#     The feature vector is the combination of all the words.
# 
# Next steps:
#     
#     We will try and improve the word vector quality.(tfidf)
#     We will try and bring in some sentence structure/contexct information (N Grams)
#     
# 

# ### The 2nd approach to text vectorization: TFIDF
# ### For example, when a 100-word document contains the term “cat” 12 times, the TF for the word ‘cat’ is
# ### TFcat = 12/100 i.e. 0.12
# ### Let’s say the size of the corpus is 1000 documents. If we assume there are 30 documents that contain the term “cat”, then
# ### IDF (cat) = log (1000/30) = 1.52
# ### (TF*IDF) cat = 0.12 * 1.52 = 0.182
# if appears in one document only then its ,1.2*log(1000/1)=3.6

# In[131]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer(min_df=20 ,max_df=.8, preprocessor= clean)
X = vectorizer.fit_transform(dat.title_clean.values)
y= dat.labels.values

Xtr, Xtt, Ytr, Ytt= train_test_split(X, y, test_size=.3, stratify= y, random_state=98)


clf = LogisticRegression(random_state=0).fit(Xtr, Ytr)

y_pred= clf.predict(Xtt)

print(accuracy_score(Ytt, y_pred)*100)
print(f1_score(Ytt, y_pred, average='macro')*100)


# In[134]:


pd.DataFrame(X[0:300].todense(), columns= vectorizer.get_feature_names())


# ### imporove it more, bring some sentence structure using N-grams
# 
# sentence1= "man bites dog"
# sentence2= "dog bites man"
# Features extracted will be:['man', 'bites', 'dog']
# 
# the sentence vector is same using above methods!
# 
# using 2-gram
# 
# Features extracted will be:
# sentence1= ['man', 'bites', 'dog', 'man bites', 'bites dog']
# sentence1= ['dog', 'bites', 'man', 'dog bites', 'bites man']
# 
# the sentence vectors are pretty different and can convey quite a bit of information as well!
# 
# 

# In[133]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer(min_df=20 ,max_df=0.8, preprocessor= clean,
                             ngram_range=(1,3))
X = vectorizer.fit_transform(dat.description)
y= dat.labels.values

Xtr, Xtt, Ytr, Ytt= train_test_split(X, y, test_size=.3, stratify= y, random_state=98)


clf = LogisticRegression(random_state=0).fit(Xtr, Ytr)

y_pred= clf.predict(Xtt)

print(accuracy_score(Ytt, y_pred)*100)
print(f1_score(Ytt, y_pred, average='macro')*100)


# In[135]:


pd.DataFrame(X[0:300].todense(), columns= vectorizer.get_feature_names())


# ### we have built our text classification model 2
# ### we did some improvements to the word vectors by using tfidf and we induced some sentence structure using n grams!

# In[136]:


### How can we improve more?

### - We have to imrove the sentence vector so that it conveys more information
### - To imrpove the sentence vector we have to improve the word vectors.

## Lets leave tfidf and move towards word embeddings


# ### Glove Embeddings
# GloVe stands for global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrix from a corpus. The resulting embeddings show interesting linear substructures of the word in vector space.

# In[137]:


embeddings_index = dict()
f = open(r'C:\Users\ELECTROBOT\PycharmProjects\bot\tarun_nlp\glove.6B.50d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# ### Going back to our cat example.
# "Cat" appears in our 100 word document 12 times
# And in 10,000,000 million documents, 0.3 million documents contain the term “cat”
# 
# Count vector = 12
# tfidf= 1.52
# 
# Glove embedding is below and it is global.

# In[138]:


embeddings_index['cat']


# In[139]:


#all all embeddings to create sentence vector

sentence= "this is a test statement"

feature_vec=np.zeros(50)


for word in sentence.split():
    feature_vec+=embeddings_index[word]
feature_vec


# In[140]:


all_vec=[]
for sent in dat.description:
    sent= sent.lower()
    sent= re.sub(r'[^a-z ]', " ", sent)
    sent= re.sub(r'\s+', " ", sent)
    sent= " ".join([i for i in sent.split() if i not in stopwords])
    sent_vec=np.zeros(50)
    counter=1
    #print(sent)
    for word in sent.split():
        try:
            sent_vec+=embeddings_index[word]
            counter+=1
        except:
            pass
    all_vec.append(sent_vec/counter)


# In[141]:


all_vec=np.vstack(all_vec)


# In[142]:


Xtr, Xtt, Ytr, Ytt= train_test_split(all_vec, y, test_size=.3, stratify= y, random_state=98)


clf = LogisticRegression(random_state=0).fit(Xtr, Ytr)

y_pred= clf.predict(Xtt)

print(accuracy_score(Ytt, y_pred)*100)
print(f1_score(Ytt, y_pred, average='macro')*100)


# In[143]:


all_vec


# In[ ]:





# In[ ]:




