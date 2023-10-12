import pandas as pd
import numpy as np
import spacy
import re
import fitz

nlp= spacy.load("en_core_web_sm")


from spacy.matcher import Matcher
matcher= Matcher(nlp.vocab)

def is_passive(sentence):
    doc= nlp(sentence)
    passive_rule=[{"DEP":"nsubjpass"}, {"DEP":"aux", "OP":"*"}, {"DEP":"auxpass"}, {"TAG":"VBN"}]
    matcher.add("Passive", None, passive_rule)
    matches= matcher(doc)
    if matches:
        return "Passive"
    else:
        return "Active"



def highlight(doc, confusing_words, empty_phrases, while_list):
    for page in doc:
        for word in confusing_words:
            word= " "+word+" "
            text_instances= page.searchFor(word)
            if text_instances!=[]:
                for inst in text_instances:
                    highlight=page.addHighlightAnnot(inst)
        for word in empty_phrases:
            text_instances= page.searchFor(word)
            if text_instances!=[]:
                for inst in text_instances:
                    highlight=page.addStrikeoutAnnot(inst)
        


        text= page.getText()
        text_b= split_into_sentences(text)
        pas=[]
        for sentence in text_b:
            sent_clean= clean(sentence)
            sent_clean= re.sub("[\(\[].*?[\)\]]", "", sent_clean)
            if is_passive(sent_clean)=="Passive":
                pas.append(sentence)

        for sent in pas:
            text_instances= page.searchFor(sent)
            if len(text_instances)==0:
                text_instances= page.searchFor(sent[0:30])
            for inst in text_instances:
                highlight= page.addUnderlineAnnot(inst)
            
        
        for sentence in text_b:
            full_form_given= re.findall(r'\([A-Z]{2,}\)', sentence)
            for ff in full_form_given:
                full_form= re.findall(f'(.*){ff}',sentence)[0][0]
                abbreviation= re.findfall(r'(?<=\().+?(?=\))',ff)[0]

                abbreviation_created= "".join([i for i in full_form if i.isupper()=True])

                if abbreviation_created[-len(abbreviation):]==abbreviation:
                    while_list.append(abbreviation)
        
        mytext_= re.findall(r'\b[A-Z]{2,}\b', text)
        mytext_= [i for i in my_text_ if i is not in while_list]

        for word in mytext_:
            word= " "+word+" "
            text_instances= page.searchFor(word)
            for inst in text_instances:
                highlight=page.addSquigglyAnnot(inst)

    return doc




