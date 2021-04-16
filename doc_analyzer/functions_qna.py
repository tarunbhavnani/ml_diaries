# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:23:17 2021

@author: ELECTROBOT
"""

import torch
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')
#model.save_pretrained("model/")


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')
#tokenizer.save_pretrained("model/")


# =============================================================================
# get the text as text_blob
# =============================================================================


# =============================================================================
# break to meaningful sentences
# =============================================================================
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|No)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|co|in)"
stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
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


all_sents= split_into_sentences(text_blob) 

# =============================================================================
# get the question as question 
# =============================================================================

question= "who did roger federer marry?"
question= "where was federer born?"
question= "what was his first win?"
question= re.sub(r'[^A-Za-z0-9 ]', " ", question)
question= re.sub(r'\s+', " ", question.strip())


# =============================================================================
# model one to get the closest using tfidf and ngram3 
# =============================================================================
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


#def get_closest(question, all_sents):

#define vectorizer
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
vectorizer.fit(all_sents)
#gectorize all sentences
tf_idf_matrix= vectorizer.transform(all_sents)

#vectorize question
question_tfidf= " ".join([i for i in question.split() if i not in stopwords])
question_vec = vectorizer.transform([question_tfidf])


#get similarities
scores=cosine_similarity(tf_idf_matrix ,question_vec)
scores=[i[0] for i in scores]
dict_scores={i:j for i,j in enumerate(scores)}
dict_scores={k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse= True)}


#get top n sentences
final_responses=[all_sents[i] for i in dict_scores]
response_sents=final_responses[0:10]

# =============================================================================
# Final answer using language model 
# =============================================================================

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    
    # ======== Tokenize ========
    
    encoded_dict = tokenizer.encode_plus(text=question,text_pair=answer_text, add_special=True)
    #encoded_dict = tokenizer.encode_plus(text=question,text_pair=answer_text, add_special=True, max_length=1024, truncation=True)
    
    
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = encoded_dict['input_ids']

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    segment_ids = encoded_dict['token_type_ids']

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    output = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(output['start_logits'])
    start_logit= output['start_logits'][0][answer_start].detach().numpy()
    answer_end = torch.argmax(output['end_logits'])

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')
    return answer, start_logit


max_logit=0
logits=[]
for num, answer_text in enumerate(response_sents):
    answer, start_logit= answer_question(question, answer_text)
    logits.append(start_logit)
    if start_logit>max_logit:
        max_logit=start_logit
        correct_answer=answer
        answer_extracted= answer_text
        answer_num=num

print(correct_answer)
print(answer_extracted)

# =============================================================================
# 
# =============================================================================
