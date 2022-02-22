# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:09:09 2022

@author: ELECTROBOT
"""


text="""Since We(SW) will train the model we will set it to train. IA(In Addition), we will run the model on the device defined above (if the cuda is available we will train the model on the GPU, otherwise on the CPU).

We will also define an optimizer. An optimizer is an object that will Help us Update the Model(HUM). There is a whole list of optimizers, but in this example we will use the AdamW optimizer which is currently one of the most popular optimizers used in deep learning.

Every optimizer accepts the model parameters (params) which can be acquired with model.parameters(). In addition, the optimizer accepts other inputs such as the learning rate lr used to specify for how much we want to move the model parameters in the direction of the gradient."""

import re

def get_abb(text):
    final_abb=[]
    full_form_given= re.findall(r'[A-Z]{2,}\(.*\)',text)
    for txt in full_form_given:
        full_form= re.findall(r'(?<=\().+?(?=\))',txt)[0]
        abbreviation_created= "".join([i[0] for i in full_form.split() if i[0].isupper()==True])
        abbreviation= txt.split("(")[0]
        if abbreviation==abbreviation_created:
            final_abb.append(abbreviation)

	#test2
    full_form_given= re.findall(r'.*\([A-Z]{2,}\)', text)
    for txt in full_form_given:
        full_form= txt.split("(")[0]
        abbreviation= re.findall(r'(?<=\().+?(?=\))',txt)[0]
        abbreviation_created= "".join([i[0] for i in full_form.split() if i[0].isupper()==True])
        if abbreviation_created[-len(abbreviation):]==abbreviation:
            final_abb.append(abbreviation)
    return final_abb



#next iteration for United States Dollar(USD)

abb=[]
all_Sents= text.split('.') #use better get rid of \n

for sentence in all_Sents:
    sentence= re.sub("\n", " ",sentence )
    #get all abbreviations which are in brackets
    full_form_given= re.findall(r'\([A-Z]{2,}\)', sentence)

    for ff in full_form_given:
        full_form= re.findall(f'(.*){ff}',sentence)[0][0]
        abbreviation= re.findall(r'(?<=\().+?(?=\))',ff)[0]
        abbreviation_created= "".join([i for i in full_form if i.isupper()==True])
        if abbreviation_created[-len(abbreviation):]==abbreviation:
            print(sentence)
            abb.append(abbreviation)













