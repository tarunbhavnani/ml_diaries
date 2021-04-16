# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:16:28 2021

@author: ELECTROBOT
"""

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
    

    def split_into_sentences(self,text):
        alphabets= "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr|No)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov|co|in)"
        stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

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
        
    
    def vectorize_text(self,text):
        all_sents= self.split_into_sentences(text)
        vec = TfidfVectorizer(min_df=1, analyzer=qnatb.ngrams)
        
        self.vectorizer=vec.fit(all_sents)
        
        self.tfidf_matrix= vec.transform(all_sents)
        
        
    def get_response_sents(self, top=10):
        question=self.question
        #vectorize question
        stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
        
        question_tfidf= " ".join([i for i in question.split() if i not in stopwords])
        question_vec = vectorizer.transform([question_tfidf])
        scores=cosine_similarity(self.tfidf_matrix ,question_vec)
        scores=[i[0] for i in scores]
        dict_scores={i:j for i,j in enumerate(scores)}
        dict_scores={k: v for k, v in sorted(dict_scores.items(), key=lambda item: item[1], reverse= True)}
        
        #get top n sentences
        final_responses=[all_sents[i] for i in dict_scores]
        response_sents=final_responses[0:top]
        
        return response_sents
        
    
    
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
    
    
    def retrieve_answer(self,question, top=10):
        
        question= re.sub(r'[^A-Za-z0-9 ]', " ", question)
        question= re.sub(r'\s+', " ", question.strip())
        self.question=question
        
        
        response_sents=self.get_response_sents(top=top)
        max_logit=3
        logits=[]
        correct_answer="Please rephrase"
        answer_extracted= "Please rephrase"

        for num, answer_text in enumerate(response_sents):
            answer, start_logit= self.answer_question(answer_text)
            logits.append(start_logit)
            if start_logit>max_logit:
                max_logit=start_logit
                correct_answer=answer
                answer_extracted= answer_text
                #answer_num=num
            
        
        return correct_answer, answer_extracted, max_logit, logits



        
        
        

# =============================================================================
#         
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



correct_answer, answer_extracted, max_logit, logits=qna.retrieve_answer(question, top=5)







        
        
        

