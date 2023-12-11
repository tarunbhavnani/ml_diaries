

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:43:35 2022

@author: ELECTROBOT
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import numpy as np
import _pickle as pickle

import os
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#def get_sentiment(sent, model, tokenizer):
#    encoded_input = tokenizer(sent, return_tensors='pt')
#    output = model(**encoded_input)
#    scores = output[0][0].detach().numpy().astype("float64")
#    neg, neutral, pos = softmax(scores)
#
#    return {"negative": neg, "neutral": neutral, "positive": pos}

def get_sentiment(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return {"negative": sentiment_dict['neg'], "neutral": sentiment_dict['neu'], "positive": sentiment_dict['pos']}

class Filetb(object):

    def __init__(self):
        self.files = None
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

        for i in range(len(words) - 1):
            if words[i] not in self.stopwords and words[i + 1] not in self.stopwords:
                bigram = f'{words[i]} {words[i + 1]}'
                ngrams.append(bigram)

        return ngrams

    @staticmethod
    def clean(sent):
        sent = re.sub(r'<.*?>|h\s*t\s*t\s*p\s*s?://\S+|www\.\S+', " ", sent)  # html tags and urls
        sent = re.sub('\n|\(.*?\)|\[.*?\]', " ", sent)  # newlines and content inside parentheses and brackets
        sent = re.sub(r'[^A-Za-z0-9\.,\?\(\)\[\]\/ ]', " ", sent)
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

    def files_processor_tb(self, files):
        self.files = files
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

def allowed_file(file):
    allowed_ext = [".pdf"]
    return True in [file.endswith(i) for i in allowed_ext]


def process_upload_files(files,upload_dir="uploads"):

    for file in files:
        if file and allowed_file(file.filename):
            try:
                #obj= file.file.read()
                file_path = os.path.join(upload_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(file.file.read())
            except Exception as e:
                print(str(e))

    names= [os.path.join(upload_dir, i) for i in os.listdir(upload_dir) if i.endswith(".pdf")]
    #names= [name.decode() for name in names]
    #names= [name for name in names if name[:len(collection_var)]==collection_var and name[-4:]==".pdf"]

    fp= Filetb()
    _= fp.files_processor_tb(names)
    #obj= pickle.dumps(fp)
    #cache.write_to_cache_user(collection_var + '_fp', obj)
    with open(os.path.join(upload_dir, "qna"), "wb") as handle:
        pickle.dump(fp, handle)

    return names




# def process_upload_files1(files):
#     user_folder = os.path.join(app.config['UPLOAD_FOLDER'], get_user_name())

#     if not os.path.isdir(user_folder):
#         os.mkdir(user_folder)

#     for file in files:
#         if file and allowed_file(file.filename):
#             try:
#                 filename = secure_filename(file.filename)
#                 file.save(os.path.join(user_folder, filename))
#             except Exception as e:
#                 print(f"Error occurred while processing file: {e}")

#     names = [os.path.join(user_folder, i) for i in os.listdir(user_folder) if i.endswith(".pdf")]

#     fp = Filetb()
#     fp.files_processor_tb(names)

#     with open(os.path.join(user_folder, "qna"), "wb") as handle:
#         pickle.dump(fp, handle)

#     return



def read_qna(upload_dir="uploads", file="qna"):
    file_path = os.path.join(upload_dir, file)

    # Read the pickled file
    with open(file_path, "rb") as handle:
        loaded_data = pickle.load(handle)
    
    return loaded_data




def reg_ind(words,upload_dir="uploads"):
    
    loaded_data= read_qna(upload_dir, file="qna")
    tb_index=loaded_data.tb_index
    if "," in words:

        words = [i.strip().lower() for i in words.split(",")]
        reg = "|".join(words)
        tb_index_reg = tb_index
        tb_index_reg = [i for i in tb_index if len(re.findall(reg, i['sentence'].lower())) > 0]

    elif "+" in words:
        words = [i.strip().lower() for i in words.split("+")]
        tb_index_reg = tb_index
        for word in words:
            tb_index_reg = [i for i in tb_index_reg if len(re.findall(word, i['sentence'].lower())) > 0]
    else:
        words = words.strip().lower()
        tb_index_reg = [i for i in tb_index if len(re.findall(words, i['sentence'].lower())) > 0]

    docs = list(set([i['doc'] for i in tb_index_reg]))

    overall_dict = {i: sum([1 for j in tb_index_reg if j['doc'] == i]) for i in docs}

    return tb_index_reg, overall_dict, docs




# =============================================================================
# check 
# =============================================================================


# import glob
# files= glob.glob(r"C:\Users\tarun\Desktop\Books\*.pdf")

# #names=process_upload_files(files,upload_dir=r"C:\Users\tarun\Desktop\check")
# fp= Filetb()
# _= fp.files_processor_tb(files)
# #obj= pickle.dumps(fp)
# #cache.write_to_cache_user(collection_var + '_fp', obj)
# import os
# upload_dir=r"C:\Users\tarun\Desktop\check"
# with open(os.path.join(upload_dir, "qna"), "wb") as handle:
#     pickle.dump(fp, handle)
    
    

# loaded_data=read_qna(upload_dir, file="qna")


# tb_index= loaded_data.tb_index

# words= "data science"
# hj= reg_ind(words,upload_dir)



