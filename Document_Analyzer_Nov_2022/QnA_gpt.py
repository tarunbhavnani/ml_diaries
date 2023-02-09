# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:24:21 2023

@author: ELECTROBOT
"""

import torch
from transformers import AutoModelForQuestionAnswering,  AutoTokenizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import numpy as np


class qnatb(object):
    

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tb_index = None
        self.all_sents = None 
        self.vec = None 
        self.tfidf_matrix = None
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                     'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                     'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                     'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                     'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                     'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                     'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                     'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                     'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                     'now']
        
        
    # def ngrams(self, string):
    #     # Remove punctuations and 'BD' from the string
    #     string = re.sub(r'[,-./]|\sBD', r'', string)
    #     ngram = []
    #     # Split the string into words
    #     words = string.split()
    #     for word in words:
    #         if word not in self.stopwords and len(word) > 3:
    #             # Generate n-grams with a length of 3 or greater
    #             for i in range(3):
    #                 nw = word[:len(word) - i]
    #                 if len(nw) > 2:
    #                     ngram.append(nw)
    #         elif word not in self.stopwords:
    #             ngram.append(word)
    #     return ngram
    
    def ngrams(self, string):
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

        return ngrams

    
    @staticmethod
    def clean(sent):
        sent = re.sub(r'<.*?>|h\s*t\s*t\s*p\s*s?://\S+|www\.\S+', " ", sent)  # html tags and urls
        sent = re.sub('\n|\(.*?\)|\[.*?\]', " ", sent)  # newlines and content inside parentheses and brackets
        sent = re.sub(r'[^A-Za-z0-9\.,\?\(\)\[\]\/ ]', " ", sent)
        sent = re.sub('\s+', " ", sent)
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
        starts = ["The", "Them", "Their", 'What', 'How', 'Why', 'Where', 'When', 'Who', 'Whom', 'Whose', 'Which', 'Whether',
                  'Can', 'Could', 'May', 'Might', 'Must', 'Shall', 'Should', 'Will', 'Would', 'Do', 'Does', 'Did', 'Has',
                  'Have', 'Had', 'Is', 'Are', 'Was', 'Were', 'Am', 'Be', 'Being', 'Been', 'If', 'Then', 'Else', 'Whether',
                  'Because', 'Since', 'So', 'Although', 'Despite', 'Until', 'While']
        
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

        #sentences = [s.strip() for s in sentences]
        final = []
        temp = ""
        for sent in sentences:
            temp += sent.strip() + " "
            if len(temp.split()) > 100:
                final.append(temp)
                temp = ""

        return final
    
    
    def files_processor_tb(self, files):
        tb_index = []
        all_sents = []
        #unread=[]
        for file in files:
            try:
                doc = fitz.open(file)
                for num, page in enumerate(doc):
                    try:
                        text = page.get_text().encode('utf8').decode('utf8')
                        text = qnatb.clean(text)
                        sentences = qnatb.split_into_sentences(text)
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
                #print(file)
                #unread.append(file)
                pass
                
        self.tb_index = tb_index
        self.all_sents = all_sents
        vec = TfidfVectorizer(analyzer= self.ngrams, lowercase=True)
        vec.fit(all_sents)
        self.vec = vec
        tfidf_matrix = vec.transform(all_sents)
        self.tfidf_matrix = tfidf_matrix
    
        return tb_index, all_sents, vec, tfidf_matrix
    
    def answer_question(self, question, answer_text):
        #encoded_dict = self.tokenizer.encode_plus(text=question, text_pair=answer_text, add_special=True)
        encoded_dict = self.tokenizer.encode_plus(text=question, text_pair=answer_text)
        input_ids = torch.tensor([encoded_dict['input_ids']])
        segment_ids = torch.tensor([encoded_dict['token_type_ids']])
    
        output = self.model(input_ids, token_type_ids=segment_ids)
    
        answer_start = torch.argmax(output['start_logits'][0])
        start_logit = output['start_logits'][0][answer_start].detach().numpy()
        answer_end = torch.argmax(output['end_logits'][0])
    
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        answer = ' '.join(tokens[answer_start:answer_end + 1]).replace(' ##', '')
        if answer[0]=="[":
            answer=""
            start_logit=0
            
    
        return answer, start_logit
    

    def get_response_sents(self, question, max_length=None):

        # Prepare the question for vectorization by removing stopwords
        question_tfidf = " ".join([word for word in question.split() if word not in self.stopwords])
        question_vec = self.vec.transform([question_tfidf])
    
        # Calculate cosine similarity scores between the question and the sentences
        scores = cosine_similarity(self.tfidf_matrix, question_vec)
        scores = [score[0] for score in scores]
    
        # Sort the sentences by their scores in descending order
        dict_scores = {index: score for index, score in enumerate(scores)}
        dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
    
        # Select the top n sentences with scores greater than 0.1
        final_response_dict = [self.tb_index[index] for index, score in dict_scores.items() if score > 0.1]
    
        # Filter out sentences with fewer than `max_length` words if specified
        if max_length:
            final_response_dict = [sent for sent in final_response_dict if len(sent['sentence'].split()) > max_length]
        
        # Use the first sentence to answer the question
        answer = self.answer_question(question, final_response_dict[0]["sentence"])
    
        return final_response_dict, answer
    
    def get_response_cosine(self, question, min_length=7, score_threshold=0.1):
        
        question= qnatb.clean(question)
        question= re.sub(r'[^a-z0-9 ]', " ", question.lower())
        
        question_tfidf = " ".join([i for i in question.split() if i not in self.stopwords])
        question_vec = self.vec.transform([question_tfidf])
        scores = cosine_similarity(self.tfidf_matrix, question_vec).flatten()
        relevant_sentences = [self.tb_index[i] for i in np.argsort(scores)[::-1] if scores[i] > score_threshold]
        final_response_dict = [sent for sent in relevant_sentences if len(sent['sentence'].split()) >= min_length]
        return final_response_dict
    
    def get_top_n(self, question, top=10, lm=True):
        
        response_sents = self.get_response_cosine(question)
        if not lm:
            return response_sents
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
    
# =============================================================================
# 
# =============================================================================

# qna= qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')
# 
# import glob
# files=glob.glob(r"C:\Users\ELECTROBOT\Desktop\data\*")
# 
# fp= qna.files_processor_tb(files)
# jk=qna.tb_index
# question="who married federer"
# final_response_dict, answer= qna.get_response_sents(question=question,max_length=10)


# responses=qna.get_top_n(question)









