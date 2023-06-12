# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:06:34 2023

@author: tarun
https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
"""
def split_into_sentences(text):
        # Define regular expressions for various patterns to be replaced
        alphabets = "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr|No)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        decimals = "(\d*[.]\d*)"
        websites = "[.](com|net|org|io|gov|co|in)"
        decimals = r'\d\.\d'
        starts = ["A","An","The", "Them", "Their", 'What', 'How', 'Why', 'Where', 'When', 'Who', 'Whom', 'Whose', 'Which',
                  'Whether',
                  'Can', 'Could', 'May', 'Might', 'Must', 'Shall', 'Should', 'Will', 'Would', 'Do', 'Does', 'Did',
                  'Has',
                  'Have', 'Had', 'Is', 'Are', 'Was', 'Were', 'Am', 'Be', 'Being', 'Been', 'If', 'Then', 'Else',
                  'Whether',
                  'Because', 'Since', 'So', 'Although', 'Despite', 'Until', 'While', "For", "We", "About"]

        http = r'h\s*t\s*t\s*p\s*s?://\S+|www\.\S+'

        # Clean and prepare the text
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(decimals, lambda g: re.sub(r'\.', '<prd>', g[0]), text)
        text = re.sub(http, lambda g: re.sub(r'\.', '<prd>', g[0]), text)
        # text= re.sub(r'(?<=\[).+?(?=\])', "", text) # remove everything inside square brackets
        # text= re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
        text = re.sub(r'\[(\w*)\]', "", text)  # remove evrything in sq brackets with sq brackets
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        if "i.e." in text: text = text.replace("i.e.", "i<prd>e<prd>")
        if "e.g" in text: text = text.replace("e.g", "e<prd>g")
        if "www." in text: text = text.replace("www.", "www<prd>")
        text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
        text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        for i in starts:
            text = text.replace("{}".format(i), ".<stop>{}".format(i))
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        # sentences = sentences[:-1]

        # sentences = [s.strip() for s in sentences]
        final = []
        temp = ""
        for sent in sentences:
            if len(sent) > 10:
                sent = re.sub(r'\d+\.(\d+)', '', sent)
                temp += sent.strip() + " "
                if len(temp.split()) > 200:
                    final.append(temp)
                    temp = ""
            else:
                pass

        return final


# =============================================================================


import spacy
import re
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_md")
#nlp = spacy.load("de_core_news_sm")

import pandas as pd


import fitz

doc=fitz.open(r"C:\Users\tarun\Desktop\notes\NLP_stanford.pdf")

text=[]
for page in doc:

    text.append(page.get_text().encode('utf8').decode('utf-8').replace('\n', " "))

text=[i[37:] for i in text]


text=" ".join(text)

sentences= split_into_sentences(text)


# =============================================================================

import spacy
nlp= spacy.load('en_core_web_md')

nlp.pipe_names
nlp.remove_pipe('tok2vec')

def get_ner(doc) -> pd.DataFrame:
    ner_list = [(word.text, word.label_) for word in doc.ents]
    temp = pd.DataFrame(ner_list, columns=["text", "ent"])
    temp = temp.groupby("ent")["text"].apply(list).reset_index().T.reset_index(drop=True)
    temp.columns=temp.iloc[0]
    temp=temp.iloc[1:]
    temp["text"] = doc.text
    temp["words"]= len(doc.text.split())
    temp['len']=len(doc.text)
    return temp

df= pd.DataFrame()

for sent in sentences:
    print('.', end=" ", flush=True)
    doc= nlp(sent)
    temp= get_ner(doc)
    df=df.append(temp)







