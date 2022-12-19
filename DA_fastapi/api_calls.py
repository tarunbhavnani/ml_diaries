# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:49:10 2022

@author: ELECTROBOT
"""
# =============================================================================
# import pickle
#     #load file processor object
# with open(r"C:\Users\ELECTROBOT\Desktop\DA_fastapi\uploads\fpp", 'rb') as handle:
#     fp = pickle.load(handle)
#     # Predicting the Class
# final_response_dict = get_response_fuzz(question="who is the yakabakazoo of federer",
#                                         vec=fp.vec,
#                                         tfidf_matrix=fp.tfidf_matrix,
#                                         tb_index=fp.tb_index,
#                                         stopwords=fp.stopwords,
#                                         max_length=7)
# 
# LM_final = qna.get_top_n("who is the yakabakazoo of federer", final_response_dict, top=10, max_length=None)
# qna.retrieve_answer( question="who is the yakabakazoo of federer", get_response_sents=final_response_dict, top=10, max_length=None)
# 
# 
# # Return the Result
# return {'class': LM_final[0]["answer"]}
# 
# 
# 
# =============================================================================
import requests

url = "http://127.0.0.1:8000/files"

payload={}
files=[
  ('files',('rf.pdf',open('C:/Users/ELECTROBOT/Desktop/data/rf.pdf','rb'),'application/pdf'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)



import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = json.dumps({
  "text": "tarun ne kya kiya"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)




import requests

url = "http://127.0.0.1:8000/reset"

payload={}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)