# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:35:59 2021

@author: tarun
"""
import os
import re
import fitz
import spacy
import pandas as pd
nlp= spacy.load('en_core_web_sm')
from operator import itemgetter
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself','they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that','these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through','during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how','all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should','now']

def valid(sentences):
    valid_sentences=[]
    for sent in sentences:
        sent1=nlp(sent)
        if [i for i in sent1.noun_chunks]!=[]:
            if [i.pos_ for i in sent1 if i.pos_=='VERB']!=[]:
                valid_sentences.append(sent)
    return valid_sentences

def valid_toggle(sent):
    case=[0 if len(re.findall(r'[A-Z]',i))==0 else 1 for i in sent.split()]
    score= sum(case)/len(case)
    return score

def load_model(model_path):
	model= BartForConditionalGeneration.from_pretrained(model_path)
	tokenizer= BartTokenizer.from_pretrained(model_path)
	return model, tokenizer

def summarize(text_blob, tokenizer, model, min_len=150, max_len=300):
	inputs= tokenizer([text_blob], max_length=1024, return_tensors='pt')
	summary_ids= model.generate(inputs['input_ids'], num_beams=3, max_length=max_len, min_length=min_len,
                             early_stopping=True)
	summary=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
	return summary[:summary.rfind('.')+1]


def fonts(doc, granularity=False):
    """Extracts fonts and their usage in PDF documents.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}

    for page in doc:
        blocks = page.getText("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles







def font_tags(font_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.
    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict
    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag 
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag



def headers_para(doc, size_tag):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict
    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    for page in doc:
        blocks = page.getText("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text

                # REMEMBER: multiple fonts and sizes are possible IN one block

                block_string = ""  # text found in block
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if s['text'].strip():  # removing whitespaces:
                            if first:
                                previous_s = s
                                first = False
                                block_string = size_tag[s['size']] + s['text']
                            else:
                                if s['size'] == previous_s['size']:

                                    if block_string and all((c == "|") for c in block_string):
                                        # block_string only contains pipes
                                        block_string = size_tag[s['size']] + s['text']
                                    if block_string == "":
                                        # new block has started, so append size tag
                                        block_string = size_tag[s['size']] + s['text']
                                    else:  # in the same block, so concatenate strings
                                        block_string += " " + s['text']

                                else:
                                    header_para.append(block_string)
                                    block_string = size_tag[s['size']] + s['text']

                                previous_s = s

                    # new block started, indicating with a pipe
                    block_string += "|"

                header_para.append(block_string)

    return header_para

def remove_p(text):
    if text[0:2]=='<p':
        text= text[3:]
    return text

def clean_bar(text):
    text= re.sub(r'\|', "", text)
    text= re.sub(r'- ', "", text)
    return text


def get_headers_paragraph(doc):
    
    font_counts, styles= fonts(doc, granularity=False)
    size_tag= font_tags(font_counts, styles)
    headers_paragraph= headers_para(doc, size_tag)
    
    headers_paragraph= [i for i in headers_paragraph if i[0:2] in ['<p', '<h']]
    
    headers_paragraph=[remove_p(i) for i in headers_paragraph]
    headers_paragraph=[clean_bar(i) for i in headers_paragraph]
    
    return headers_paragraph

    
def get_all_sents(doc):
    headers_paragraph=get_headers_paragraph(doc)
    temp=""
    final=[]
    for sent in headers_paragraph:
        if sent[0:2]!="<h":
            if sent.strip()[-1]!=".":
                temp+=" "
                temp+=sent
            else:
                temp+=" "
                temp+=sent
                final.append(temp)
                temp=""
    final.append(temp)
    return final


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
    text= re.sub(r'(?<=\[).+?(?=\])', "", text) # remove everything inside square brackets
    text= re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
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

    
    
def clean_junk(text):
    text=re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*', "", text)
    text=re.sub(r'http.*', " ", text)
    text=re.sub('<[^<]+?>', '', text)
    text=re.sub(r'(?<=\[).+?(?=\])', "", text) 
    text=re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
    text=re.sub(r'[^A-Za-z0-9 /., ]', " ", text)
    text=re.sub(r'\s+', " ", text)
    return text
    
    

    
    
def get_weighted_summary_pdf(doc):
    #doc=fitz.open(os.path.join(folder, file))
    all_sentences=get_all_sents(doc)
    all_sentences=[clean_junk(i) for i in all_sentences]
    
    words={}
    for sent in all_sentences:
        for word in sent.lower().split():
            if word not in stopwords:
                if word not in words:
                    words[word]=1
                else:
                    words[word]+=1
    
    maxi=max([j for i,j in words.items()])


    weighted_freq=pd.DataFrame(words.items(), columns=['word', "freq"])
    weighted_freq['wt']=[i/maxi for i in weighted_freq.freq]
    
    wt_fr={i:j for i,j in zip(weighted_freq.word, weighted_freq.wt)}
    
    def weight(sent):
       
        return sum([wt_fr[token] if token in wt_fr else 0 for token in sent.lower().split()])
       
    sent_weights=[weight(sent) for sent in all_sentences]
    
    fdf= pd.DataFrame({"sentence": all_sentences, "weights":sent_weights})
    
    #thresh=fdf['weights'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    for quant in reversed([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
        thresh=fdf['weights'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])[quant]
        if len(" ".join([i for i in fdf[fdf.weights>thresh].sentence]).split())>1000:
            break
    
    
    fdf1= fdf[fdf.weights>thresh]
    
    text_blob=" ".join([i for i in fdf1.sentence])
    return text_blob
    


def custom_sentencizer(doc):
    for i, token in enumerate(doc[0:-2]):
        if token.pos_ in ['DET', 'PRON', 'INTJ','ADP'] and doc[i].is_title:
            doc[i].is_sent_start=True
        if token.text in [":", "(", ">", "-", ",", ";"]:
            doc[1+1].is_sent_start=False
            doc[i].is_sent_start=False
        if len(re.findall(r'[A-z/(/)]', token.text[0]))==1:
            doc[i].is_sent_start=False
        if token.text=='o':
            doc[1+1].is_sent_start =True
            
    return doc


def nlp_sent():
    sbd= nlp.create_pipe('sentencizer')
    nlp.add_pipe(sbd, before="parser")
    nlp.add_pipe(custom_sentencizer, after='sentencizer')
    print(nlp.pipe_names)
    return nlp


# =============================================================================
# new spacy
# =============================================================================

#from spacy.language import Language
#nlp = spacy.load("en_core_web_sm", exclude=['tok2vec','ner','attribute_ruler', 'lemmatizer' ])
#print(nlp.pipe_names)


#nlp.add_pipe("info_component", name="custom_sentencizer", last=True)

# @Language.component("custom_sentencizer")
# def custom_sentencizer(doc):
#     for i, token in enumerate(doc[0:-2]):
#         if token.pos_ in ['DET', 'PRON', 'INTJ','ADP'] and doc[i].is_title:
            
#             doc[i].is_sent_start=True
#         if token.text in [":", "(", ">", "-", ",", ";"]:
#             doc[1+1].is_sent_start=False
#             doc[i].is_sent_start=False
#         if len(re.findall(r'[a-z/(/)]', token.text[0]))==1:
#             doc[i].is_sent_start=False
#         if token.text=='o':
#             doc[1+1].is_sent_start =True
            
#     return doc


# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("custom_sentencizer", before="parser")  # Insert before the parser
# #nlp=nlp_sent()

# "this is . the final. says who"
# import re
# doc = nlp("This is. A sentence.  This is. Another sentence.")
# for sent in doc.sents:
#     print(sent.text)

# =============================================================================
# 
# =============================================================================





