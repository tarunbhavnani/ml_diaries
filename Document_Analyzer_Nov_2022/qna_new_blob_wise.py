# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:24:21 2023

@author: ELECTROBOT
"""

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import numpy as np

class Filetb(object):

    def __init__(self):
        self.files=None
        self.tb_index = None
        self.all_sents = None
        self.vec = None
        self.tfidf_matrix = None
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                          'yourself',
                          'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                          'itself',
                          'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                          'that',
                          'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                          'had',
                          'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                          'as',
                          'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                          'through',
                          'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                          'off',
                          'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                          'how',
                          'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                          'not',
                          'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                          'should',
                          'now']


    def ngrams(self,string):
        cleaned_string = re.sub(r'[,-./]|\sBD', '', string)
        ngrams = []
        words = cleaned_string.split()
        for word in words:
            if word not in self.stopwords and len(word) > 3:
                # dont break  words that contain numbers
                if not any(char.isdigit() for char in word):
                    # Create n-grams by taking substrings of the word
                    for i in range(3):
                        ngram = word[:len(word) - i]
                        if len(ngram) > 2:
                            ngrams.append(ngram)
                else:
                    ngrams.append(word)
            elif word not in self.stopwords:
                ngrams.append(word)
        
        for i in range(len(words) - 1):
            if words[i] not in self.stopwords and words[i + 1] not in self.stopwords:
                bigram = f'{words[i]} {words[i + 1]}'
                ngrams.append(bigram)
        
        
        return ngrams

    @staticmethod
    def clean(sent):
        sent = re.sub(r'<.*?>|h\s*t\s*t\s*p\s*s?://\S+|www\.\S+', " ", sent)  # html tags and urls
        sent = re.sub('\n|\(.*?\)|\[.*?\]', " ", sent)  # newlines and content inside parentheses and brackets
        #sent = re.sub(r'[^A-Za-z0-9\.,\?\(\)\[\]\/ ]', " ", sent)
        sent = re.sub(r"\.+", ".", sent)
        sent = re.sub('\s+', " ", sent)
        sent = re.sub(r'For internal use only \d{1,2} (of|\/) \d{1,2}', " ", sent)
        return sent
    
        
        
    
    @staticmethod
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
        starts = ["The", "Them", "Their", 'What', 'How', 'Why', 'Where', 'When', 'Who', 'Whom', 'Whose', 'Which',
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
        
        
        sentences=[ re.sub(r'\d+\.(\d+)', '', i) for i in sentences if len(i)>20]
        sentences=[ i.strip() for i in sentences]
        
        return sentences
        
 

    def files_processor_tb(self, files):
        self.files=files
        tb_index = []
        all_sents = []
        # unread=[]
        for file in self.files:
            try:
                doc = fitz.open(file)
                for num, page in enumerate(doc):
                    try:
                        text = page.get_text().encode('utf8').decode('utf8')
                        text = Filetb.clean(text)
                        sentences = Filetb.split_into_sentences(text)
                        for sent in sentences:
                            tb_index.append({
                                "doc": file.split('\\')[-1],
                                "page": num,
                                "sentence": sent
                            })
                            all_sents.append(sent.lower())
                    except:
                        tb_index.append({
                            "doc": file.split('\\')[-1],
                            "page": num,
                            "sentence": ""
                        })
            except:
                # print(file)
                # unread.append(file)
                pass

        self.tb_index = tb_index
        self.all_sents = all_sents
        vec = TfidfVectorizer(analyzer=self.ngrams, lowercase=True)
        vec.fit(all_sents)
        self.vec = vec
        tfidf_matrix = vec.transform(all_sents)
        self.tfidf_matrix = tfidf_matrix

        return tb_index, all_sents, vec, tfidf_matrix
    
    def get_response_cosine(self, question, min_length=7, score_threshold=0.1):

        question = Filetb.clean(question)
        question = re.sub(r'[^a-z0-9 ]', " ", question.lower())

        question_tfidf = " ".join([i for i in question.split() if i not in self.stopwords])
        question_vec = self.vec.transform([question_tfidf])
        scores = cosine_similarity(self.tfidf_matrix, question_vec).flatten()
        relevant_sentences = [self.tb_index[i] for i in np.argsort(scores)[::-1] if scores[i] > score_threshold]
        final_response_dict = [sent for sent in relevant_sentences if len(sent['sentence'].split()) >= min_length]
        return final_response_dict


    


class Qnatb(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def answer_question(self, question, answer_text, model="minilm"):
        question = re.sub(r"[^a-z0-9 ]", " ", question)
        question = re.sub(r"\s+", " ", question)
        question = question.strip() + " ?"
        if model=="bert":
            encoded_dict = self.tokenizer.encode_plus(text=question, text_pair=answer_text,  truncation=True, max_length=512)
        
        elif model=="minilm":
            encoded_dict = self.tokenizer.encode_plus(text=question, text_pair=answer_text, truncation=True, max_length=512)
        input_ids = torch.tensor([encoded_dict['input_ids']])
        segment_ids = torch.tensor([encoded_dict['token_type_ids']])

        output = self.model(input_ids, token_type_ids=segment_ids)

        answer_start = torch.argmax(output['start_logits'][0])
        start_logit = float(output['start_logits'][0][answer_start].detach().numpy())
        answer_end = torch.argmax(output['end_logits'][0])

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        answer = ' '.join(tokens[answer_start:answer_end + 1]).replace(' ##', '')
        if "[" in answer:
            answer = ""
            start_logit = 0

        return answer, start_logit


    def get_top_n(self, question,response_sents, top=10):
        top_responses = []
        for num, answer_text in enumerate(response_sents[:top]):
            answer, start_logit = self.answer_question(question, answer_text['sentence'])
            top_response = {}
            top_response = response_sents[num].copy()
            top_response['start_logit'] = start_logit
            top_response['answer'] = answer
            top_responses.append(top_response)
        top_responses = sorted(top_responses, key=lambda item: item['start_logit'], reverse=True)
        return top_responses + response_sents[top:]


    def extract_answer_blobs(self, question, responses, model="minilm"):
        
        #responses= self.get_response_cosine(question)
        
        final=[]
        temp=[]
        for i in responses:
            if len(i["sentence"].split())>200:
                final.append(i["sentence"])
            else:
                temp.append(i["sentence"])
                if len(" ".join(i for i in temp).split())>200:
                    final.append(" ".join(i for i in temp))
                    temp=[]
                    #break
                else:
                    pass
        
        final.append(" ".join(i for i in temp))
        
        result=[]
        for sentence in final:
            answer, start_logit=self.answer_question(question, sentence, model=model)
            result.append((answer, start_logit, sentence))
        
        result=sorted(result, key=lambda x:x[1])[::-1]
        return result[0]
            
        
        
        
    
# =============================================================================
#
# =============================================================================

# qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')
# qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\Bert-qa')
#
# import glob
# files_path=r"C:\Users\ELECTROBOT\Desktop\data\*"
# files=glob.glob(files_path)

# %%time
# fp= Filetb()
# fp.files_processor_tb(files)
# jk=fp.tb_index


qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')
#question="who married federer"
#question="when did federer win his first wimbledon"
question="when did federer win his first us open"
responses= fp.get_response_cosine(question)
results=qna.extract_answer_blobs(question, responses[:50])


qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\Bert-qa\model')
question="who married federer"
question="when did federer win his first wimbledon"
question="when did federer win his first us open"
responses= fp.get_response_cosine(question)
%%time
results=qna.extract_answer_blobs(question, responses[:50], model="bert")



#%%time
#responses=qna.get_top_n(question=question,response_sents=responses, top=10)