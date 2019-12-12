#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:11:43 2019

@author: tarun.bhavnani
"""

#pick columns by name regex


def regex_col(dat, reg, inout=1):
    if inout==1:
        dat=dat[[i for i in dat if i in list(dat.filter(regex=reg))]]
        return dat
    elif inout==2:
        dat=dat[[i for i in dat if i not in list(dat.filter(regex=reg))]]
        return dat
    else:
        print("error in inout value")
        return dat



def null_analysis(dat):
    null_col=pd.DataFrame(dat.isnull().sum())
    null_col.plot(title="cols")
    null_row=pd.DataFrame(dat.isnull().sum(axis=1))
    null_row.plot(title="rows")
    
    return null_col, null_row


#null_analysis(train)
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



def find_columns(dat, col, drop=0):
    if drop=0:
        dat_req=dat[list(dat.filter(regex=i))]
        return dat_req
    elif drop =1:
        dat_req= dat.drop(list(dat.filter(regex=i)), axis=1)
        return dat_req
    else:
        print("error: drop value")


        


##########################################################################################
#create tdm
##########################################################################################



jk=[" ".join([re.sub(" ","_",re.sub("'","",re.sub("-"," ",j)).strip()) for j in i]) for i in file1['rr']]


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#docs = ['why hello there', 'omg hello pony', 'she went there? omg']
vec = CountVectorizer()
X = vec.fit_transform(jk)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df)
words= list(df)

df['data']=jk
df['application_status']=file1['application_status']


##########################################################################################
#wordcloud
##########################################################################################


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    #plt.savefig(fname=os.path.join(os.getcwd(), "{}.png".format(i)))
    plt.show()
    plt.imsave(os.path.join(os.getcwd(), "{}.png".format(i)), wordcloud)

show_wordcloud(df['data'].iloc[1:100])
 
 
status= set(df['application_status'])

for i in status:
    show_wordcloud(df['data'][df['application_status']==i])
    

