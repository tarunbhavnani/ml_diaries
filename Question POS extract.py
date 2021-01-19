import re
import nltk

import pandas as pd
item="Fingerprinting is required  they must be present for their appointments. o If you enrolled during our June or July events  please remember to add your Known Traveller Number  KTN  to your travel profile in the Concur online booking tool."
item="are there?, is that?"
item="What documents are needed to update my traveler profile in the future ?"
#this one not managed yet
item="Please advise how do we proceed from here ? , How do I do this ?, â€? Will there be any issue once I submit the form ?, Hello , How are you ?"
item="What is the process to get the Visa renewed here ?, Why has this issue come up suddenly ?"
item="can you do that? how do we proceed? where you do that?, what can I do"
item="Ana receives an error message when she tries to get to the AMEX homepage."

item="""How was your AskHR experience?, Do you know where it is?,Do you know where it is?,
Have we checked the June data?,are you managing this communication? """
def proct(item):
    
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
	    #for i in range(len(tagged)):
            # #print(i)
            #  dit[tagged[i][0]]=tagged[i][1]
  
            #dit.items()
            #dit.keys()
            #dit.values()
            
	    #   ('word'), ('tag')
            #entities = re.findall(r'\(\'(\w*)\',\s\'W\w?\w?\'',str(tagged))
            
            #see here it is w** or M** using pipe.
            #entities =re.findall(r'\(\'(\w*)\',\s\'[W|M]\w?\w?\'\),\s\(\'(\w*)\',\s\'P\w?\w?\'',str(tagged))
            
            #entities =re.findall(r'\(\'(\w*)\',\s\'[W|M]\w?\w?\'\),\s\(\'(\w*)\',\s\'P\w?\w?\'\),\s\(\'(\w*)\',\s\'V\w?\w?\'\)',str(tagged))
            entities1 =re.findall(r'\(\'(\w*)\',\s\'M\w?\w?\'\),\s\(\'(\w*)\',\s\'[P|R]\w?\w?\'\),\s\(\'(\w*)\',\s\'V\w?\w?\'\)',str(tagged))
            entities2 =re.findall(r'\(\'(\w*)\',\s\'W\w?\w?\'\),\s\(\'(\w*)\',\s\'M\w?\w?\'\)',str(tagged))
            entities3 =re.findall(r'\(\'(\w*)\',\s\'VBP\'\),\s\(\'(\w*)\',\s\'PRP\'\),\s\(\'(\w*)\',\s\'V\w?\w?\'\)',str(tagged))
            entities4 =re.findall(r'\(\'(\w*)\',\s\'W\w?\w?\'\),\s\(\'(\w*)\',\s\'V\w?\w?\'\),\s\(\'(\w*)\',\s\'[D|P]\w?\w?\'\)',str(tagged))
            #entities5 =re.findall(r'\(\'(\w*)\',\s\'V\w?\w?\'\),\s\(\'(\w*)\',\s\'[R|I]\w?\w?\'\)',str(tagged))
            
            #see broken to understand, this is three words like "how could you"
            #entities = re.findall(r'
            #\(\'(\w*)\',\s\'M\w?\w?\'\)#('word', 'POS')
            #,\s                        # comma space
            #\(\'(\w*)\',\s\'P\w?\w?\'\)#('word', 'POS')
            #,\s                        #comma space  
            #\(\'(\w*)\',\s\'V\w?\w?\'\)',str(tagged))
        
            #su=len(entities1)+len(entities2)+len(entities3)+len(entities4)+len(entities5)
            #ln=len(entities) +len(entities2)
            
            
            
            if len(entities1) >= 1:
                a=1
            else:
                if len(entities2)>0:
                    a=1
                else:
                    if len(entities3)>0:
                        a=1
                    else: 
                        if len(entities4)>0:
                            a=1
                        else:
                            a=0
                
            return(a)


#len(tagged)
#entities = re.findall(r'\w+',str(tagged))
#len(entities)
#entities =re.findall(r'\(\'(\w*)\',\s\'[W|V]\w?\w?\'\),\s\(\'(\w*)\',\s\'P\w?\w?\'',str(tagged))
#item="""If you consider this to be a reasonable business expense
#  please ensure you have approval from your manager before registering for this event."""

#for k in ty.text:
#  print(proct(k))
ty=pd.read_csv("C:\\Users\\v-tabhav\\Desktop\\final_tok.csv",encoding='latin-1')

#check and remove nulls in 'text'    

ty[ty.text.isnull()]

ty.text[ty['text'].isnull()]="nothing here"

ty['check']=ty['text'].apply(proct)
ty['check'].value_counts()

tyq=ty[ty['check']==1]

ty.to_csv("tyq_qs23.csv", sep = ',', index = False)

import os
os.chdir("C:\\Users\\v-tabhav\\Desktop\\Targetibility\\")
os.getcwd()
#df["hjk"]=df["Loan_Status"].map({'Y':1,'N':0})

#ty.check=0
#op="what are you doing why toiday?"

#processContent()
		



#re.findall(r'<W..?>*',str(tagged))

#ty.ko = re.sub("[^a-zA-Z]",str(ty.text))



#Kannan's Method!
def isq(verbatim):
    qtags=['WDT','WP', 'WPS', 'WRB','MD']
    Q=0
    for sentence in nltk.sent_tokenize(verbatim):
        for tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
            if any (qt.lower()==tag[1].lower() for qt in qtags):
                Q=1
    return Q

isq("I am here what the fk are u doing there")
isq("ypu can always come back")            
item="""what is your name?, how should we proceed?, Can you help me?, How do I register?,
How should I regfister?"""





#check tags:

cht=" he she you them they I "
nltk.pos_tag(nltk.word_tokenize(cht))


#if a qtags=['WDT','WP', 'WPS', 'WRB','MD'] comes before a PRP its a question.


grammer=r'{(<JJ>* <NN.*>+ <IN>)? '

re.findall(r'\(\'(\w*)\',\s\'N\w?\w?\'\),\s\(\'(\w*)\',\s\'V\w?\w?\'',str(tagged))





dt=['art', 'automatic keyphrase extraction', 'changes in topic', 'concise description',
'content', 'coverage', 'difficulty', 'document', 'document categorization',
'document length', 'extraction of important topical words', 'fundamental difficulty',
'general rule of thumb', 'gold standard', 'good coverage', 'human-labeled keyphrases',
'humans', 'indexing', 'keyphrases', 'major topics', 'many other core nlp tasks',
'much research', 'natural language processing for purposes', 'particular knowledge domains',
'phrases from documents', 'search', 'semantic similarity with other documents',
'set of keyphrases', 'several factors', 'state', 'structural inconsistency',
'summarization', 'survey', 'terminology extraction', 'topics', 'wide applicability', 'work']

#?nltk.regexpParser()

chunker = nltk.chunk.regexp.RegexpParser(grammar)

chunker.parse(nltk.pos_tag(dt))

st="""whois that #i  am walking.
      #i was walking."""
re.findall(r'[\w\s\.-]*#[\w\s\.-]*', st)

re.findall(r'#[\w\s\.-]+', st)
re.findall(r'#[\w\s\.-]*', st)
           
           

list=["a cat has a cat follower", "come on cat a life"]
regex=re.compile(".*(cat).*")
regex.search("a cat has a cat follower, come on live a life").group(1)
[m.group(0) for l in list for m in [regex.search(l)] if m]

           

re.match(r'[\w\s]*', st)
re.findall(r'\'|w+\'', dt)

#try and extract the full question.
entities35 =re.findall(r'\(\'(\w*)\',\s\'VBP\'\),\s\(\'(\w*)\',\s\'PRP\'\),\s\(\'(\w*)\',\s\'V\w?\w?\'\)(.*)\s\(\'(\w*)\',\s\'N\w?\w?\'\)',str(tagged))
entities35