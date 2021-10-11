# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:57:48 2021

@author: ELECTROBOT
"""
import requests


# =============================================================================
# test file upload
# =============================================================================
# url ='http://127.0.0.1:5000/upload_static_file'
# files = {'files[]': open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\Arria123.pdf", 'rb')}
# response = requests.post(url, files=files)

with open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\Arria123.pdf", 'rb') as f:
    response = requests.post('http://127.0.0.1:5000/upload_static_file', files={'files[]': f})


with open(r"C:\Users\ELECTROBOT\Desktop\1629725512502.jpg", 'rb') as f:
    response = requests.post('http://127.0.0.1:5000/upload_static_file', files={'files[]': f})


response.status_code
response.headers
response.content
response.text
response.json()
jsonApiData = response.json()

# =============================================================================
# sessions
# =============================================================================




session = requests.Session()
session.headers.update({'Authorization': 'Bearer {access_token}'})
#response = session.get('http://127.0.0.1:5000')

with open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\Arria123.pdf", 'rb') as f:
    response = session.post('http://127.0.0.1:5000/upload_static_file', files={'files[]': f})

response.text

#search

response = session.post('http://127.0.0.1:5000/search', data = {'search':'what is arria'})


jsonApiData = response.json()



response = session.get('http://127.0.0.1:5000/metadata/Arria123.pdf/')


jsonApiData = response.json()













