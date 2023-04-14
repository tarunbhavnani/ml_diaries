# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:12:28 2021

@author: tarun
"""
import re
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

def valid(sentences):
    valid_sentences=[]
    for sent in sentences:
        sent1=nlp(sent)
        if [i for i in sent1.noun_chunks]!=[]:
            if [i.pos_ for i in sent1 if i.pos_=='VERB']!=[]:
                valid_sentences.append(sent)
    return valid_sentences

def valid_toggle(sent):
    case=[0 if len(re.findall(r'[A-Z]',i))==0 else 1 for i in sent.split()]
    score= sum(case)/len(case)
    return score

def load_model(model_path):
	model= BartForConditionalGeneration.from_pretrained(model_path)
	tokenizer= BartTokenizer.from_pretrained(model_path)
	return model, tokenizer

def summarize(text_blob, tokenizer, model, min_len=150, max_len=300):
	inputs= tokenizer([text_blob], max_length=1024, return_tensors='pt')
	summary_ids= model.generate(inputs['input_ids'], num_beams=3, max_length=max_len, min_length=min_len,
                             early_stopping=True)
	summary=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
	return summary[:summary.rfind('.')+1]


model_path=r'C:\Users\tarun\Desktop\summarization_bart\model_files'
model, tokenizer= load_model(model_path)



text_blob= """Roger Federer (German: [ˈrɔdʒər ˈfeːdərər]; born 8 August 1981) is a Swiss professional tennis player. He is ranked No. 9 in the world by the Association of Tennis Professionals (ATP). He has won 20 Grand Slam men's singles titles, an all-time record shared with Rafael Nadal and Novak Djokovic. Federer has been world No. 1 in the ATP rankings a total of 310 weeks – including a record 237 consecutive weeks – and has finished as the year-end No. 1 five times. Federer has won 103 ATP singles titles, the second-most of all-time behind Jimmy Connors and including a record six ATP Finals.

Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time.[c] A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009.

Although Federer remained in the top 3 through most of the 2010s, the success of Djokovic and Nadal in particular ended his dominance over grass and hard courts. From mid-2010 through the end of 2016, he only won one major title. During this period, Federer and Stan Wawrinka led the Switzerland Davis Cup team to their first title in 2014, adding to the gold medal they won together in doubles at the 2008 Beijing Olympics. Federer also has a silver medal in singles from the 2012 London Olympics, where he finished runner-up to Andy Murray. After taking half a year off in late 2016 to recover from knee surgery, Federer had a renaissance at the majors. He won three more Grand Slam singles titles over the next two years, including the 2017 Australian Open over Nadal and a men's singles record eighth Wimbledon title later in 2017. He also became the oldest ATP world No. 1 in 2018 at age 36.

A versatile all-court player, Federer's perceived effortlessness has made him highly popular among tennis fans. Originally lacking self-control as a junior, Federer transformed his on-court demeanor to become well-liked for his general graciousness, winning the Stefan Edberg Sportsmanship Award 13 times. He has also won the Laureus World Sportsman of the Year award a record five times. Outside of competing, he played an instrumental role in the creation of the Laver Cup team competition. Federer is also an active philanthropist. He established the Roger Federer Foundation, which targets impoverished children in southern Africa, and has raised funds in part through the Match for Africa exhibition series. Federer is routinely one of the top ten highest-paid athletes in any sport, and ranked first among all athletes with $100 million in endorsement income in 2020.[4]"""



text_blob=re.sub(r'(?<=\[).+(?=\])', "", text_blob)
text_blob=re.sub(r'(?<=\().+(?=\))', "", text_blob)
text_blob= re.sub(r'[^A-za-z0-9. ]', " ", text_blob)
text_blob= re.sub(r'\s+', " ", text_blob)


if len(text_blob)<150:
	summary= summarize(text_blob, tokenizer, model, min_len=0, max_len= len(text_blob))
else:
	summary= summarize(text_blob, tokenizer, model)
