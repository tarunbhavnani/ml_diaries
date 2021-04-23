# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:16:28 2021

@author: ELECTROBOT

1.85 to 1|DOT|85
re.sub(r'\d*\.\d*', lambda g: re.sub(r'\.', '|DOT|', g[0]), tt)
"""
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz

class qnatb(object):
    
    def __init__(self, model_path):
        self.model_path= model_path
        self.model = BertForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
    @staticmethod
    def ngrams(string, n=3):
        string = re.sub(r'[,-./]|\sBD',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]
    
    @staticmethod
    def clean(sent):
        sent= re.sub(r'<.*?>', " ", sent)#html tags
        sent=re.sub(r'h\s*t\s*t\s*p\s*s?://\S+|www\.\S+', " ", sent) #remove urls
        
        sent=re.sub('\n', " ", sent) 
        sent=re.sub(r'\(.*?\)', " ", sent)#all inside ()
        sent=re.sub(r'\[.*?\]', " ", sent)#all inside ()
        sent= re.sub(r'[^A-Za-z0-9\.,\?\(\)\[\]\/ ]', " ", sent)
        sent=re.sub('\s+', " ", sent) 
        return sent

    @staticmethod
    def split_into_sentences(text):
        alphabets= "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr|No)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        decimals =  "(\d*[.]\d*)"
        websites = "[.](com|net|org|io|gov|co|in)"
        decimals =  r'\d\.\d'
        stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

        text = " " + text + "  "
        text = text.replace("\n"," ")
        text= re.sub(decimals, lambda g: re.sub(r'\.', '<prd>', g[0]), text)
        #text= re.sub(r'(?<=\[).+?(?=\])', "", text) # remove everything inside square brackets
        #text= re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
        text=re.sub(r'\[(\w*)\]', "", text)# remove evrything in sq brackets with sq brackets
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>")
        if "e.g" in text: text = text.replace("e.g","e<prd>g")
        if "www." in text: text = text.replace("www.","www<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        #sentences = sentences[:-1]
        
        sentences = [s.strip() for s in sentences]
        return sentences
        
    
    def vectorize_text(self,text):
        text= self.clean(text)
        self.all_sents= qnatb.split_into_sentences(text)
        vec = TfidfVectorizer(min_df=1, analyzer=qnatb.ngrams)
        
        self.vectorizer=vec.fit(self.all_sents)
        
        self.tfidf_matrix= vec.transform(self.all_sents)
        
    
    def vectorize_question(self,question):
        question= re.sub(r'[^A-Za-z0-9 ]', " ", question)
        question= re.sub(r'\s+', " ", question.strip())
        self.question=question
        


    def get_response_sents(self):
        question=self.question
        #vectorize question
        stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
        
        question_tfidf= " ".join([i for i in question.split() if i not in stopwords])
        question_vec = self.vectorizer.transform([question_tfidf])
        scores=cosine_similarity(self.tfidf_matrix ,question_vec)
        scores=[i[0] for i in scores]
        dict_scores={i:j for i,j in enumerate(scores)}
        dict_scores={k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse= True)}
        
        #get top n sentences
        final_responses=[self.all_sents[i] for i in dict_scores]
        final_responses= [i for i in final_responses if len(i.split())>3]
        #response_sents=final_responses[0:top]
        
        return final_responses
        
    
    
    def answer_question(self, answer_text):
    
        encoded_dict = self.tokenizer.encode_plus(text=self.question,text_pair=answer_text, add_special=True)
        input_ids = encoded_dict['input_ids']
        segment_ids = encoded_dict['token_type_ids']
        assert len(segment_ids) == len(input_ids)
        output = self.model(torch.tensor([input_ids]), # The tokens representing our input text.
                                        token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
    
        answer_start = torch.argmax(output['start_logits'])
        start_logit= output['start_logits'][0][answer_start].detach().numpy()
        answer_end = torch.argmax(output['end_logits'])
    
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            else:
                answer += ' ' + tokens[i]
        return answer, start_logit
    
    
    def retrieve_answer(self, top=10):
                
        
        response_sents=self.get_response_sents()
        max_logit=3
        logits=[]
        correct_answer="Please rephrase"
        answer_extracted= "Please rephrase"

        for num, answer_text in enumerate(response_sents[0:top]):
            answer, start_logit= self.answer_question(answer_text)
            logits.append(start_logit)
            if start_logit>max_logit:
                max_logit=start_logit
                correct_answer=answer
                answer_extracted= answer_text
                #answer_num=num
            
        
        return correct_answer, answer_extracted, max_logit, logits



        
# =============================================================================
# function to read multiple files and search and give results with filename and page number
# =============================================================================


def read_pdf(path):
    doc = fitz.open(path)
    file={}
    for num, page in enumerate(doc):
        try:
            text=page.getText().encode('utf8')
            text= text.decode('utf8')
            file[num]=text
        except:
            file[num]=""
    return file
        
text= read_pdf(path="C:/Users/ELECTROBOT/Desktop/rf.pdf")


text1= [qnatb.clean(i) for i in text.values()]
text2=" ".join([i for i in text1])
all_sents= split_into_sentences(text2)

text_blob=" ".join([i for i in text.values()])

text_blob= re.sub(r'<.*?>', " ", text_blob)#html tags
text_blob=re.sub(r'h\s*t\s*t\s*p\s*s?://\S+|www\.\S+', " ", text_blob) #remove urls
text_blob= re.sub(r'<.*?>', " ", text_blob)#html tags
text_blob=re.sub('\n', " ", text_blob) #remove urls
text_blob=re.sub(r'\(.*?\)', " ", text_blob)
text_blob=re.sub('\s+', " ", text_blob) #remove urls




#text_blob= qna.split_into_sentences(text_blob)
#responses=sim_tb(text_blob, "who is roger federer married to?", top=10)

    

3def search(string, file):
    
    
    
# =============================================================================
# not workig because of wrong sentences, try and improve the similarity sentences 

"USE BERT EMBEDDING/TOKENIZER"
"iMPROVE LOADING QUESTION AND TEXT"

# =============================================================================


        

# =============================================================================
#define a better clean, remove all https etc
# =============================================================================
        
qna= qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')

#induce text

qna.vectorize_text(text_blob)


#get answer
question="who is roger federer married to?"
question="who is roger federers wife?"
question="who is roger federer?"
question="when did roger federer win his fiorst wimbeldon"
question="how many grandslams has federer won"
question="who is federers father"
question="who is federers mother"


qna.vectorize_question(question)

correct_answer, answer_extracted, max_logit, logits=qna.retrieve_answer( top=15)


responses=qna.get_response_sents()
responses[0:15]

file = read_pdf("C:/Users/ELECTROBOT/Desktop/output.pdf")
    
        


#what if the vectorizer is set on question??

        
        
        

