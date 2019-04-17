#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:33:57 2019

@author: tarun.bhavnani@dev.smecorner.com
Information extraction
https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag


ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

sent=word_tokenize(ex)
sent= pos_tag(sent)

#we get tuples of words with their respective pos tags. But the funny tihng with pos tags is that 
#they change for the same words depending on the occurance of the word in a sentence.

#Chunking to extract ners

"Lets create a chunk NP for extracting noun phrases"

pattern="NP:{<DT>?<JJ>*<NN>}"

#lets test it

cp= nltk.RegexpParser(pattern)
cs= cp.parse(sent)
print(cs)
#we can se the NPs identified


"IOB tags are used for this alot, we will also"

from nltk.chunk import conllstr2tree, tree2conlltags
from pprint import pprint

iob_tags=tree2conlltags(cs)
pprint(iob_tags)





ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)
#you can see e google is a NNP but it is categorized as a person, thats bad!!


#lets do the same by using spacy!!
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp= en_core_web_sm.load()

ex='European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

doc= nlp(ex)
"This does everything, now we just have to extract things from doc"
doc

pprint([(X.text, X.label_) for X in doc.ents])
"""#hey google gets categorized as an org!!
#European is NORD (nationalities or religious or political groups), Google is an organization,
 $5.1 billion is monetary value and Wednesday is a date object. """


"""The BILOU tagging scheme

B- Begin
I- Inner token
L- Last token
U- Single token entity
O- Non-entity token

"""

pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])



##############
##############


from bs4 import BeautifulSoup
import requests
import re
def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))


#url="https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769"
url='https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news'

ny_bb = url_to_string(url)

article = nlp(ny_bb)

number_of_entities=len(article.ents)

labels = [x.label_ for x in article.ents]
pprint(Counter(labels))
len(Counter(labels))

"There are {} entites with a total of {} different unique labels".format(number_of_entities, len(Counter(labels)))

#Most frequent tokens:

items = [x.text for x in article.ents]
Counter(items).most_common(3)


#randomly selecting one sentence:

sentences = [x for x in article.sents]
print(sentences[20])


#Let’s run displacy.render to generate the raw markup.
#in jupyter notebook

displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')
displacy.render(nlp(str(sentences)), jupyter=True, style='ent')
























