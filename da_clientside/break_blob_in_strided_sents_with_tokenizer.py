# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:34:09 2022

@author: ELECTROBOT
"""

ed = tokenizer(jk.tolist(),
                    is_split_into_words=True,
                    return_overflowing_tokens=True,
                    stride=DOC_STRIDE,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                    truncation=True)

ed.overflow_to_sample_mapping


len(jk.tolist()[1])#646
len(ed.input_ids[1]) + len([i for i in ed.input_ids[2] if i !=1]) #919

tokenizer.decode(ed.input_ids[1])
tokenizer.decode(ed.input_ids[2])

import re
def split_into_sentences(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|No)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    decimals = "(\d*[.]\d*)"
    websites = "[.](com|net|org|io|gov|co|in)"
    decimals = r'\d\.\d'

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(decimals, lambda g: re.sub(r'\.', '<prd>', g[0]), text)
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
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    # sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]
    return sentences
def clean(sent):
    sent = re.sub(r'<.*?>', " ", sent)  # html tags
    sent = re.sub(r'h\s*t\s*t\s*p\s*s?://\S+|www\.\S+', " ", sent)  # remove urls

    sent = re.sub('\n', " ", sent)
    sent = re.sub(r'\(.*?\)', " ", sent)  # all inside ()
    sent = re.sub(r'\[.*?\]', " ", sent)  # all inside ()
    sent = re.sub(r'[^A-Za-z0-9\.,\?\(\)\[\]\/ ]', " ", sent)
    sent = re.sub('\s+', " ", sent)
    return sent
import fitz
doc = fitz.open(r"C:\Users\ELECTROBOT\Desktop\rf.pdf")

rf=""
for num, page in enumerate(doc):
    try:
        text = page.getText().encode('utf8')
        text = text.decode('utf8')
        text = clean(text)
        rf+=" "
        rf+=text
            
    except:
        pass

rf_split= rf.split()

tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)

ed = tokenizer(rf_split,
                    is_split_into_words=True,
                    return_overflowing_tokens=True,
                    stride=25,
                    max_length=150,
                    padding="max_length",
                    truncation=True)

len(ed.input_ids)
tokenizer.decode(ed.input_ids[0])
list(ed)




