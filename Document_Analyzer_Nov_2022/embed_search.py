# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:32:09 2022

@author: ELECTROBOT
"""
import numpy as np
import re
import glob
files=glob.glob(r"C:\Users\ELECTROBOT\Desktop\data\*")

fp= file_processor(files)

tb= fp.tb_index
stopwords=fp.stopwords


#get embeddings

# load the whole embedding into memory
embeddings_index = dict()
f = open(r"C:\Users\ELECTROBOT\Desktop\model_dump\glove_50\glove.6B.50d.txt", encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


#embeddings_index["the"]


def clean_(sent, embeddings_index, stopwords):
    sent= re.sub(r'[^a-z ]', "", sent.lower())
    return " ".join([i for i in sent.split() if i not in stopwords and i in embeddings_index])


def get_embed(sent):
    sent= clean_(sent,embeddings_index, stopwords)
    if len(sent)>0:
        embed=sum([embeddings_index[i] for i in sent.split()])/len(sent.split())
        return embed.tolist()
    else:
        return [0]*50
    

#get embeddings for all
%%time
embed= [get_embed(i["sentence"]) for i in tb]




#get_embed(qes)

kl=dict(zip(fp.vec.get_feature_names(), fp.tfidf_matrix.toarray()[0]))



#[i for i,j in enumerate(tb) if "federer" in j["sentence"].lower()]
#Out[394]: [23288, 24655, 25652]

def tfidf_word(num_sent,word):
    
    ind=[i for i,j in enumerate(fp.vec.get_feature_names()) if j==word]
    
    return fp.tfidf_matrix.T[ind].toarray()[0][num_sent]
    
#tfidf_word(26032,"federer")
    


#[j for i,j in zip(fp.vec.get_feature_names(), fp.tfidf_matrix.toarray()[0])  if i=="federer"]

#[i for i,j in enumerate(fp.vec.get_feature_names()) if j=="married"]


#fp.tfidf_matrix[0,204682]


#get embeddings for all
embed= [get_embed(i["sentence"]) for i in tb]


qes="who is federer married to"

embed_q= np.asarray(get_embed(qes))

scores = cosine_similarity(embed, embed_q.reshape(1,-1))
scores = [i[0] for i in scores]
dict_scores = {i: j for i, j in enumerate(scores)}
dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
# get top n sentences
# final_response_dict=[self.tb_index[i] for i in dict_scores]
final_response_dict = [tb[i] for i, j in dict_scores.items() if j > 0.1]


[i for i,j in enumerate(final_response_dict) if "miroslava" in j["sentence"].lower()]


#sents_=[clean_(i["sentence"], embeddings_index, stopwords) for i in tb]

#embed=[[embeddings_index[j] for j in i.split() ] for i in sents_]




#embed=[sum([embeddings_index[j] for j in i.split()])/len(i.split()) if len(i.split())>0 else np.array([0]*50)for i in sents_]


#embed=[i.tolist() for i in embed]



#embed=[np.mean([embeddings_index[j.lower()] for j in i["sentence"].split() if j.lower() in  embeddings_index ]) for i in tb]





#sum([embeddings_index[i] for i in tb[0]["sentence"].lower().split() if i in embeddings_index])/len(tb[0]["sentence"].lower().split())


#embed=[sum([embeddings_index[i] for i in j["sentence"].lower().split() if i in embeddings_index])/len(tb[0]["sentence"].lower().split()) for j in tb]



# for i in range(len(embed)):
#     try:
#         len(embed[i])
#     except:
#         embed[i]=np.array([0]*50)





q_embed=sum([embeddings_index[i] for i in qes.lower().split() if i in embeddings_index])/len(qes.split())



scores = cosine_similarity(embed, q_embed.reshape(1,-1))
scores = [i[0] for i in scores]
dict_scores = {i: j for i, j in enumerate(scores)}
dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
# get top n sentences
# final_response_dict=[self.tb_index[i] for i in dict_scores]
final_response_dict = [tb[i] for i, j in dict_scores.items() if j > 0.1]

# final_responses=[self.all_sents[i] for i in dict_scores]



# =============================================================================
# 
# =============================================================================


def tfidf_word(num_sent,word):
    
    try:
        ind=[i for i,j in enumerate(fp.vec.get_feature_names()) if j==word]
        
        return fp.tfidf_matrix.T[ind].toarray()[0][num_sent]
    except:
        return 0


def get_embed(num,sent):
    sent= clean_(sent,embeddings_index, stopwords)
    if len(sent)>0:
        
        embed=sum([embeddings_index[i]*tfidf_word(num, i) for i in sent.split()])/len(sent.split())
        
        return embed.tolist()
    else:
        return [0]*50
    
    



%%time
embed= [get_embed(num, i["sentence"]) for num, i in enumerate(tb)]

# for num,i in enumerate(tb):
#     get_embed(num, i["sentence"])
    

# for j in sent.split():
#     embeddings_index[j]*tfidf_word(num, j)
    

# ind=[i for i,j in enumerate(fp.vec.get_feature_names()) if j=="federer"]



io=[i[1].tolist() for i in embeddings_index.items()]
io_=[i[0] for i in embeddings_index.items()]

#similar words

word= "federer"

word=embeddings_index[word]



scores = cosine_similarity(io, word.reshape(1,-1))
scores = [i[0] for i in scores]
dict_scores = {i: j for i, j in enumerate(scores)}
dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
# get top n sentences
# final_response_dict=[self.tb_index[i] for i in dict_scores]
final_response_dict = [io_[i] for i, j in dict_scores.items() if j > 0.5]

all_words= final_response_dict[0:20]


#get close words
io=[i[1].tolist() for i in embeddings_index.items()]
io_=[i[0] for i in embeddings_index.items()]

def related_(word,embeddings_index,io,io_):
    word=embeddings_index[word]
    
    scores = cosine_similarity(io, word.reshape(1,-1))
    scores = [i[0] for i in scores]
    dict_scores = {i: j for i, j in enumerate(scores)}
    dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
    # get top n sentences
    # final_response_dict=[self.tb_index[i] for i in dict_scores]
    final_response_dict = [io_[i] for i, j in dict_scores.items() if j > 0.5]

    return final_response_dict[0:20]


        


def check_w(sent, words):
    





qes="who is federer married to"

qes= clean_(qes, embeddings_index, stopwords)

all_words= [related_(i,embeddings_index,io,io_) for i in qes.split()]


all_words= sum(all_words, [])


chosen=[i for i in tb if len(re.findall("|".join([i for i in all_words]), i["sentence"]))>0]

#from here we can do fuzzy as well


embed_q= np.asarray(get_embed(qes))

embed_chosen= [get_embed( i["sentence"]) for num, i in enumerate(chosen)]

scores = cosine_similarity(embed_chosen, embed_q.reshape(1,-1))
scores = [i[0] for i in scores]
dict_scores = {i: j for i, j in enumerate(scores)}
dict_scores = {k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse=True)}
# get top n sentences
# final_response_dict=[self.tb_index[i] for i in dict_scores]
final_response_dict = [chosen[i] for i, j in dict_scores.items() if j > 0.1]












    



    














