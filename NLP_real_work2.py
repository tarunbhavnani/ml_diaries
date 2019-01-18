#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:54:06 2019

@author: tarun.bhavnani@dev.smecorner.com
http://www.nltk.org/book/ch07.html#tab-db-locations
"""
import nltk, re, pprint

locs = [('Omnicom', 'IN', 'New York'),
       ('DDB Needham', 'IN', 'New York'),
       ('Kaplan Thaler Group', 'IN', 'New York'),
       ('BBDO South', 'IN', 'Atlanta'),
       ('Georgia-Pacific', 'IN', 'Atlanta')]

query = [e1 for (e1, rel, e2) in locs if e2=='New York']
print(query)

#['BBDO South', 'Georgia-Pacific']

"#1)"

"#sentence segmentation, tokenization and pos tagging"

def ie_preprocess(document):
   sentences = nltk.sent_tokenize(document) 
   sentences = [nltk.word_tokenize(sent) for sent in sentences] 
   sentences = [nltk.pos_tag(sent) for sent in sentences] 
   return(sentences)
   
text=["tarun bhavnani","works in smecorner","is a data dcientist and an NLP scientost",
      "tarun is building a chatbot"]
[ie_preprocess(x) for x in text]
document=text

ie_preprocess(text[3])


"#2)"

"#entity detection"

#chunking will be used for entity detection

#segment and lable multi label sequences. These do not overlap


"Noun Phrase Chunking"

# a single regular-expression rule. This rule says that an NP chunk should be formed whenever
# the chunker finds an optional determiner (DT) followed by any number of adjectives (JJ) and 
#then a noun (NN). Using this grammar, we create a chunk parser, and test it on our example sentence.
# The result is a tree, which we can either print , or display graphically.



sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), 
            ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

#act_sent=" ".join([x[0] for x in sentence])


grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar) 
result = cp.parse(sentence) 
print(result) 



"Tag Patterns"
#The rules that make up a chunk grammar use tag patterns to describe sequences of tagged words. 
#A tag pattern is a sequence of part-of-speech tags delimited using angle brackets, e.g. <DT>?<JJ>*<NN>.



"#chunking with Regular Expression"


grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"), 
                 ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
 	
print(cp.parse(sentence))




nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
cp = nltk.RegexpParser(grammar)
print(cp.parse(nouns))


grammar = "NP: {<NN>+}  # Chunk consecutive nouns, one or more"
grammar = "NP: {<NN>{2,}}  # Chunk consecutive nouns, two or more"
cp = nltk.RegexpParser(grammar)
print(cp.parse(nouns))




#exploring text corpora
#we saw how we could interrogate a tagged corpus to extract phrases matching a particular 
#sequence of part-of-speech tags. We can do the same work more easily with a chunker, as follows:

cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
brown = nltk.corpus.brown
for sent in brown.tagged_sents():
  tree = cp.parse(sent)
  for subtree in tree.subtrees():
    if subtree.label() == 'CHUNK': print(subtree)
    #if subtree.label() == 'NP': print(subtree)



#representing chunks, tags vs trees

from nltk.corpus import conll2000
print(conll2000.chunked_sents('train.txt')

#evaluations

from nltk.corpus import conll2000
print(conll2000.chunked_sents('train.txt')[199])

print(conll2000.chunked_sents('train.txt', chunk_types=['NP'])[199])


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

#1)
cp = nltk.RegexpParser("")

print(cp.evaluate(test_sents))


#2)

grammar = r"NP: {<[CDJNP].*>+}"

cp = nltk.RegexpParser(grammar)

print(cp.evaluate(test_sents))

#3)

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): 
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data) 

    def parse(self, sentence): 
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


cp= UnigramChunker(train_sents)
print(cp.evaluate(test_sents))




#4)
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): 
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data) 

    def parse(self, sentence): 
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

cp=BigramChunker(train_sents)
print(cp.evaluate(test_sents))




