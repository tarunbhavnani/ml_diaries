# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:43:51 2021

@author: tarun
"""
#load docs ijn app
#get the path of the upload folder 

import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\git\ml_diaries\da_clientside\uploads\96ffac660ed3413c9b123c8c1c6d20ae')
import glob
names=glob.glob(r'C:\Users\ELECTROBOT\Desktop\git\ml_diaries\da_clientside\uploads\96ffac660ed3413c9b123c8c1c6d20ae\*')


#os.listdir()

qna=qnatb()
names=[i for i in os.listdir() if i!='qna']


Folder=r'C:\Users\ELECTROBOT\Desktop\git\ml_diaries\da_clientside\uploads\96ffac660ed3413c9b123c8c1c6d20ae'
 tb_index, all_sents, response_file_processing= qna.files_processor_tb(Folder)


tb_index, all_sents= qna.files_processor_tb(names)
tb_index, all_sents, vec, tfidf_matrix=qna.files_processor_tb(names)
#
question="who married federer"

question_tfidf = " ".join([i for i in question.split() if i not in qnatb.stopwords])

# ng=ngrams(question_tfidf)
# counter=0
# final_score=0
# for word in ng:
    
#     word_vec= vec.transform([word])
#     score = cosine_similarity(tfidf_matrix, question_vec)
#     final_score+=score
#     counter+=1
# final_score=final_score/counter
# scores = [i[0] for i in final_score]
# dict_scores = {i: j for i, j in enumerate(scores)}
# dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
# # get top n sentences
# # final_response_dict=[self.tb_index[i] for i in dict_scores]
# final_response_dict = [tb_index[i] for i, j in dict_scores.items() if j > 0.1]



question_vec = vec.transform([question_tfidf])



scores = cosine_similarity(tfidf_matrix, question_vec)
scores = [i[0] for i in scores]
dict_scores = {i: j for i, j in enumerate(scores)}
dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
# get top n sentences
# final_response_dict=[self.tb_index[i] for i in dict_scores]
final_response_dict = [tb_index[i] for i, j in dict_scores.items() if j > 0.1]


from fuzzywuzzy import fuzz
fuzz.partial_ratio("federer marry", "Roger Federer Early Life, Family Net Worth .")


question="who did federer marry"
question_tfidf = " ".join([i for i in question.split() if i not in qnatb.stopwords])

def get_score(question_tfidf, sent):
    scr=0
    counter=0
    sent = " ".join([i for i in sent.lower().split() if i not in qnatb.stopwords])
    if len(set(qna.ngrams(question_tfidf)).intersection(qna.ngrams(sent)))>3:
        for token in question_tfidf.split():
            
                scr+=fuzz.partial_ratio(token,sent)
                counter+=1
    if counter>0:
        return scr/counter
    else:
        return 0


#dict_scores={num:fuzz.partial_ratio(question_tfidf,i['sentence']) for num,i in enumerate(tb_index)}
dict_scores={num:get_score(question_tfidf,i['sentence']) for num,i in enumerate(tb_index)}



dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}

final_response_dict = [tb_index[i] for i, j in dict_scores.items()]



# =============================================================================
# ngram check with cosine similarity
# =============================================================================

qna=qnatb()
#names=[i for i in os.listdir() if i!='qna']
names=glob.glob(r'C:\Users\ELECTROBOT\Desktop\pdf_files\*')

tb_index, all_sents, vec, tfidf_matrix=qna.files_processor_tb(names)
#
question="who did federer marry"
question="who are federers parents"
question="who are federers father and mother"
question="how is heroku deployed"
question_tfidf = " ".join([i for i in question.split() if i not in qnatb.stopwords])

question_vec = vec.transform([question_tfidf])

scores = cosine_similarity(tfidf_matrix, question_vec)
scores = [i[0] for i in scores]
dict_scores = {i: j for i, j in enumerate(scores)}
dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
# get top n sentences
# final_response_dict=[self.tb_index[i] for i in dict_scores]
final_response_dict = [tb_index[i] for i, j in dict_scores.items() if j > 0.1]


#keeping only thos who have something from all

def check_ngram_intersection(question_tfidf, sentence):
    for token in question_tfidf.split():
        if len(set(qna.ngrams(token)).intersection(qna.ngrams(sentence)))==0:
            return 0
    return 1

fgh=[i for i in final_response_dict if check_ngram_intersection(question_tfidf, i['sentence'])==1]
      


def check_ngram_intersection_score(question_tfidf, sentence):
    score=0
    for token in question_tfidf.split():
        if len(set(qna.ngrams(token)[0]).intersection(qna.ngrams(sentence.lower())))>0:
            score+=1
    return score


f_dict_scores= {i:j+(check_ngram_intersection_score(question_tfidf,tb_index[i]['sentence'])) for i,j in dict_scores.items()}
f_dict_scores = {k: v for k, v in sorted(f_dict_scores.items(), key=lambda item: item[1], reverse=True)}

final_response_dict = [tb_index[i] for i, j in f_dict_scores.items() if j > 0.1]
        


#len(set(qna.ngrams(question_tfidf)).intersection(qna.ngrams(sent)))>3:
#fgh=[i for i in final_response_dict if len(set(qna.ngrams(question_tfidf)).intersection(qna.ngrams(i['sentence'])))>2]





# =============================================================================
# 
# =============================================================================

#testimg regex

reg_data= "tarun"

import pickle
qna_loaded=qna

tb_index_reg, overall_dict, docs= qna_loaded.reg_ind(reg_data)



audit_trail= session['audit_trail']
audit_trail[reg_data]=overall_dict
session['audit_trail']=audit_trail

tables= []
for doc in docs:
    try:
        cut= [i for i in tb_index_reg if i['doc']==doc]
        cut= pd.DataFrame(cut)
        pd.set_option('display.max_colwidth', 40)
        tables.append(cut.to_html(classes='data', justify='left', col_space='100px'))
    except:
        pass

    return render_template("regex.html", overall_dict=overall_dict, tables=tables, reg_data=reg_data, zip=zip)