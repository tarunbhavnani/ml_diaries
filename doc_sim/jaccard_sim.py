# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:56:00 2021

@author: ELECTROBOT
"""

def calculate_jaccard(word_tokens1, word_tokens2):
	# Combine both tokens to find union.
	both_tokens = word_tokens1 + word_tokens2
	union = set(both_tokens)

	# Calculate intersection.
	intersection = set()
	for w in word_tokens1:
		if w in word_tokens2:
			intersection.add(w)

	jaccard_score = len(intersection)/len(union)
	return jaccard_score





documents=['today india is playing against australia', 'india fought with pakistan', 'india plays']
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen=64

tok= Tokenizer(split=" ")
tok.fit_on_texts(documents)
tokenized_document=tok.texts_to_sequences(documents)


calculate_jaccard(tokenized_document[0], tokenized_document[1])