# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:05:48 2022

@author: ELECTROBOT
"""

import spacy


from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)

patterns = [
    [{'POS':'ADJ'}], [{'POS':'ADV'}],
    ]
matcher.add("demo", patterns)

doc = nlp("There is a beautiful envelope and the other one is ugly also it notoriously difficult.")


matches = matcher(doc)
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # The matched span
    print(match_id, string_id, start, end, span.text)
    
