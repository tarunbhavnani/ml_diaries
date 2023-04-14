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


import requests


session = requests.Session()
#session.headers.update({'Authorization': 'Bearer {access_token}'})
#response = session.get('http://127.0.0.1:5000')

with open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\Arria123.pdf", 'rb') as f:
    response = session.post('http://127.0.0.1:5000/upload_static_file', files={'filenames': f})

response.text

#search

response = session.post('http://127.0.0.1:5000/search', data = {'search':'what is arria'})


jsonApiData = response.json()



response = session.get('http://127.0.0.1:5000/metadata/Arria123.pdf/')


jsonApiData = response.json()










session = requests.Session()
#session.headers.update({'Authorization': 'Bearer {access_token}'})
#response = session.get('http://127.0.0.1:5000')

files = [('filenames', open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\Arria123.pdf", 'rb')),
         ('filenames', open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\pymupdf-readthedocs-io-en-latest.pdf", 'rb')),
         ('filenames', open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\HerokuTutorial.pdf",'rb'))]

#with open(r"C:\Users\ELECTROBOT\Desktop\pdf_files\Arria123.pdf", 'rb') as f:
response = session.post('http://127.0.0.1:5000/document', files=files)

response
response.json()


response=session.get('http://127.0.0.1:5000/document')
response
response.json()

response=session.get('http://127.0.0.1:5000/document/Arria123.pdf')
response
response.json()


response = session.get('http://127.0.0.1:5000/metadata/Arria123.pdf/')
response
response.json()

response = session.get('http://127.0.0.1:5000/metadata/pymupdf-readthedocs-io-en-latest.pdf/')
response
response.json()




response= session.get('http://127.0.0.1:5000/search?search=what+is+heroku')
response
response.json()


response=session.get('http://127.0.0.1:5000/delete')
response
response.json()

response=session.get('http://127.0.0.1:5000/document')
response
response.json()








curl --location --request POST 'http://127.0.0.1:5000/document' --form 'filenames=@"/C:/Users/ELECTROBOT/Desktop/pdf_files/CML_8896_17-Dec-2020_1247_1234.pdf"'


