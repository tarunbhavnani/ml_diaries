# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:46:30 2021

@author: ELECTROBOT
"""

#fast qna

#lets read all pdf from desktop
import fitz
import spacy
nlp= spacy.load(r"C:/Users/ELECTROBOT/Anaconda3/envs/bot/Lib/site-packages/en_core_web_sm/en_core_web_sm-2.3.1")

import os
os.getcwd()



# Valid sentences rules:
#     must have a noun chunk 
#     must have a verb


files={}

for filename in os.listdir('C:\\Users\\ELECTROBOT\\Desktop'):
    if filename.endswith('.pdf'):
    #if ".pdf" in filename:
        print(filename)
        
        
        try: 
            file={}
            doc=  fitz.open(os.path.join("C:\\Users\\ELECTROBOT\\Desktop",filename))
            for num, page in enumerate(doc):
                text=page.getText().encode('utf8')
                text= text.decode('utf8')
                text= split_into_sentences(text)
                sentences=[]
                for sent in text:
                    sent1= nlp(sent)
                    if [i for i in sent1.noun_chunks] !=[]:
                        if [i.pos_ for i in sent1 if i.pos_=="VERB"]!=[]:
                            sentences.append(sent)
                file[num]=sentences
                files[filename]=file
        except:
            files[filename]=[]
        

sent2= nlp(file[0][2])
[(i,i.dep_, i.pos_) for i in sent2]
