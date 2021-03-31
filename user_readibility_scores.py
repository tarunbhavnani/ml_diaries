#!/usr/bin/env python
# coding: utf-8

# 
# ### user readibility scores
# 
# #https://medium.com/analytics-vidhya/visualising-text-complexity-with-readability-formulas-c86474efc730
# 
# 
# 

# In[11]:


from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)


# In[14]:


dat_check=twenty_train['data'][0]


# In[15]:


dat_check


# In[16]:


# -*- coding: utf-8 -*-
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
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
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


# In[17]:


split= split_into_sentences(dat_check)


# In[40]:


sentences=[re.sub(r'[^A-Za-z0-9 /. ]'," ",i).strip() for i in split if len(i.split())>2]


# In[20]:


from urllib.request import urlopen


# In[27]:


link = "http://countwordsworth.com/download/DaleChallEasyWordList.txt"
f = urlopen(link)
myfile = f.read()
print(myfile)


# In[68]:


easy_words=myfile.decode('utf-8').split('\n')


# In[41]:


def ARI(text):
    score = 0.0 
    if len(text) > 0:
        score = 4.71 * (len(text) / len(text.split()) ) +  0.5 * ( len(text.split()) / len(text.split('.'))) - 21.43 
        return score if score > 0 else 0
"""
1	5-6	Kindergarten
2	6-7	1st/2nd Grade
3	7-9	3rd grade
4	9-10	4th grade
5	10-11	5th grade
6	11-12	6th grade
7	12-13	7th grade
8	13-14	8th grade
9	14-15	9th grade
10	15-16	10th grade
11	16-17	11th grade
12	17-18	12th grade
13	18-24	College student
14	24+	Professor

"""


# In[56]:


text_blob= " ".join([i for i in sentences])


# In[59]:


ARI(text_blob)


# In[65]:


len(text_blob.split('.'))


# In[55]:


sentences= ['My point is that they are doing a lot of harm on the way in the meantime.',
 'And that they converge is counterfactual  religions appear to split and    diverge.','I think  however  that women are the spiritual equals of men is clearly and unambiguously implied in the above verse  and that since women can clearly be  forgiven  and  rewarded  they  must  have souls  from the above verse .',
 'Let s try to understand what the Qur an is trying to teach  rather than try to see how many ways it can be misinterpreted by ignoring this passage or that passage.']


# In[58]:


text_blob


# In[66]:


def FleschKincaidTest(text):
	score = 0.0
	if len(text) > 0:
		score = (0.39 * len(text.split()) / len(text.split('.')) ) + 11.8 * ( sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","y"] else 0,text))) / len(text.split())) - 15.59
		return score if score > 0 else 0


# In[67]:


FleschKincaidTest(text_blob)


# In[75]:


#Dale-Chall Readability

#.1579*(diff_words/words)*100+ .0496*(words/sentences)


def dale_chall(text):
    diff_words=[i for i in text.split() if i not in easy_words]
    words=[i for i in text.split()]
    sentences= text.split('.')
    
    score= .1579*(len(diff_words)/len(words))*100+ .0496*(len(words)/len(sentences))
    
    return score

#0-5- 4th grade
#5-6- 6th
#6-7- 8th
#7-8- 10th
#8-9- 12th
#9-10- above



# In[76]:


dale_chall(text_blob)


# In[77]:


def FleschReadabilityEase(text):
	if len(text) > 0:
		return 206.835 - (1.015 * len(text.split()) / len(text.split('.')) ) - 84.6 * (sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","y"] else 0,text))) / len(text.split()))

    
    
#90-100-- 11 yr old
#60-70- 13-15 yr old
#30 or below-university graduates


# In[78]:


FleschReadabilityEase(text_blob)


# In[ ]:




