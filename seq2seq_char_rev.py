#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:51:14 2019

@author: tarun.bhavnani@dev.smecorner.com
"""


with open("fra.txt","r", encoding="Latin1") as f:
  lines=f.read().split("\n")
  
with open("fra.txt",'r', encoding='Latin1') as f:
  lines= f.read().split('\n')
  

input_texts=[]
target_texts=[]
input_characters=set()
target_characters=set()
num=10000
for i in lines[:min(num, len(lines)-1)]:
  input_text, target_text= i.split('\t')
  input_texts.append(input_text)
  target_text="\t"+target_text+"\n"
  # We use "tab" as the "start sequence" character
  # for the targets, and "\n" as "end sequence" character.
  target_texts.append(target_text)
  
  for j in input_text:
    if j not in input_characters:
      input_characters.add(j)
  for k in target_text:
    if k not in target_characters:
      target_characters.add(k)
      




input_characters= sorted(list(input_characters))
target_characters= sorted(list(target_characters))


num_encoder_tokens= len(input_characters)
num_decoder_tokens= len(target_characters)

max_encoder_seq_length= max([len(i) for i in input_texts])
max_decoder_seq_length= max([len(i) for i in target_texts])


input_token_index= dict([(j,i) for i,j in enumerate(input_characters)])
target_token_index=dict([(j,i) for i,j in enumerate(target_characters)])



#numbers of docs, max length of a doc, nom of diff occurances in doc
encoder_input_data= np.zeros((len(input_texts),max_encoder_seq_length, num_encoder_tokens), dtype="float32")
encoder_input_data.shape

decoder_input_data= np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_input_data.shape

decoder_target_data= np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data.shape



#now we have to fill these data

"basically we will create n(doc number) arrays of size[num_tokens, max_seq_length]"


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
  #print(i, input_text, target_text)
  for t,char in enumerate(input_text):
    #print(t,char)
    encoder_input_data[i,t, input_token_index[char]]=1
  for t ,char in enumerate(target_text):
    
    decoder_input_data[i,t,target_token_index[char]]=1
    if t>0:
      decoder_target_data[i,t-1, target_token_index[char]]
      
  
  
  
  
  
##data is made, now we have to process it!!


