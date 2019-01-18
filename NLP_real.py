#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:13:54 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
#https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e
import spacy

# Load the large English NLP model
nlp = spacy.load('en_core_web_lg')

# The text we want to examine
text = """London is the capital and most populous city of England and 
the United Kingdom.  Standing on the River Thames in the south east 
of the island of Great Britain, London has been a major settlement 
for two millennia. It was founded by the Romans, who named it Londinium.
"""

# Parse the text with spaCy. This runs the entire pipeline.
doc = nlp(text)

# 'doc' now contains a parsed version of text. We can use it to do anything we want!
# For example, this will print out all the named entities that were detected:
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")
    

################333
    

# Replace a token with "REDACTED" if it is a name
def replace_name_with_placeholder(token):
    if token.ent_iob != 0 and token.ent_type_ == "PERSON":
        return "[REDACTED] "
    else:
        return token.string

# Loop through all the entities in a document and check if they are names
def scrub(text):
    doc = nlp(text)
    for ent in doc.ents:
        ent.merge()
    tokens = map(replace_name_with_placeholder, doc)
    return "".join(tokens)

s = """
In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence". In 1957, Noam Chomsky’s 
Syntactic Structures revolutionized Linguistics with 'universal grammar', a rule based system of syntactic structures.
"""

print(scrub(s))


########################################


#extracting facts
import spacy
import textacy.extract

# Load the large English NLP model
#nlp = spacy.load('en_core_web_lg')

# The text we want to examine
text = """London is the capital and most populous city of England and  the United Kingdom.  
Standing on the River Thames in the south east of the island of Great Britain, 
London has been a major settlement  for two millennia.  It was founded by the Romans, 
who named it Londinium.
"""

# Parse the document with spaCy
doc = nlp(text)

# Extract semi-structured statements
statements = textacy.extract.semistructured_statements(doc, "London")

# Print the results
print("Here are the things I know about London:")

for statement in statements:
#  print(statement)
    subject, verb, fact = statement
    print(f" - {fact}")
#view raw
    

text="""s an ancient name, attested already in the first century AD, usually in the Latinised form Londinium;[62] for example, handwritten Roman tablets recovered in the city originating from AD 65/70-80 include the word Londinio ("in London").[63]

Over the years, the name has attracted many mythicising explanations. The earliest attested appears in Geoffrey of Monmouth's Historia Regum Britanniae, written around 1136.[62] This had it that the name originated from a supposed King Lud, who had allegedly taken over the city and named it Kaerlud.[64]

Modern scientific analyses of the name must account for the origins of the different forms found in early sources Latin (usually Londinium), Old English (usually Lunden), and Welsh (usually Llundein), with reference to the known developments over time of sounds in those different languages. It is agreed that the name came into these languages from Common Brythonic; recent work tends to reconstruct the lost Celtic form of the name as *[Londonjon] or something similar. This was adapted into Latin as Londinium and borrowed into Old English, the ancestor-language of English.[65]

The toponymy of the Common Brythonic form is much debated. A prominent explanation was Richard Coates's 1998 argument 
that the name derived from pre-Celtic Old European *(p)lowonida, meaning "river too wide to ford". Coates suggested that 
this was a name given to the part of the River Thames which flows through London; from this, the settlement gained the 
Celtic form of its name, *Lowonidonjon.[66] However, most work has accepted a Celtic origin for the name, and recent 
studies have favoured an explanation along the lines of a Celtic derivative of an proto-Indo-European root *lendh- 
('sink, cause to sink'), combined with the Celtic suffix *-injo- or *-onjo- (used to form place-names). Peter Schrijver
 has specifically suggested, on these grounds, that the name originally meant 'place that floods (periodically, tidally)
"""
import spacy
import textacy.extract

# Load the large English NLP model
text=[x.lower() for x in text.split()]
text=" ".join([re.sub('[^a-z]',"",x) for x in text ])
nlp = spacy.load('en_core_web_lg')

# The text we want to examine
#text = """London is [.. shortened for space ..]"""

# Parse the document with spaCy
doc = nlp(text)

# Extract noun chunks that appear
noun_chunks = textacy.extract.noun_chunks(doc, min_freq=3)

# Convert noun chunks to lowercase strings
noun_chunks = map(str, noun_chunks)
noun_chunks = map(str.lower, noun_chunks)

# Print out any nouns that are at least 2 words long
for noun_chunk in set(noun_chunks):
    if len(noun_chunk.split(" ")) > 1:
        print(noun_chunk)
        
