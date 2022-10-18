# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:50:40 2022

@author: ELECTROBOT
"""

import requests, PyPDF2, io

url = 'http://www.asx.com.au/asxpdf/20171108/pdf/43p1l61zf2yct8.pdf'
response = requests.get(url)
txt=[]

#pypdf2
# with io.BytesIO(response.content) as open_pdf_file:
#     #read_pdf= fitz.open(open_pdf_file)
#     read_pdf = PyPDF2.PdfFileReader(open_pdf_file)
#     num_pages = read_pdf.getNumPages()
#     for page in range(read_pdf.numPages):
        
#         pg=read_pdf.getPage(page)
#         text= pg.extractText()
#         txt.append(text)
    
#     print(num_pages)
    

#pymupdf
import fitz

doc=fitz.open(filename=None, stream= io.BytesIO(response.content), filetype="xref")



for num, page in enumerate(doc):
    #pass
    page.getText().encode("utf-8").decode()
