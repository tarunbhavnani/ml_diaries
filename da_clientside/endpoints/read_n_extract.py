# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 20:00:25 2021

@author: tarun
"""
import fitz
import os
import pandas as pd
import io
from PIL import Image
import numpy as np
import tabula

# do the iflese for pptx and docsx here, later append in index in qna.files_processor_tb for pptx and docx.



filename= "abc.pdf"
filename.endswith(".pdf")

class read_all(object):
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name
        if file_name.endswith(".pdf"):
            self.doctype="pdf"
            self.doc = fitz.open(os.path.join(file_path, file_name))
        elif file_name.endswith(".docx"):
            self.doctype="docx"
            self.doc= Document(os.path.join(file_path, file_name))
    
    @staticmethod
    def getMetaData_docx(doc):
        metadata = {}
        prop = doc.core_properties
        metadata["author"] = prop.author
        metadata["category"] = prop.category
        metadata["comments"] = prop.comments
        metadata["content_status"] = prop.content_status
        metadata["created"] = prop.created
        metadata["identifier"] = prop.identifier
        metadata["keywords"] = prop.keywords
        metadata["language"] = prop.language
        metadata["modified"] = prop.modified
        metadata["subject"] = prop.subject
        metadata["title"] = prop.title
        metadata["version"] = prop.version
        return metadata


    def get_metadata(self):
        #md = self.doc.metadata
        
        if self.doctype=="pdf":
            md_df = pd.DataFrame(self.doc.metadata.items(), columns=["Parameter", "Details"])
        elif self.doctype=="docx":
            md= read_all.getMetaData_docx(self.doc)
            md_df = pd.DataFrame(md.items(), columns=["Parameter", "Details"])
            
            
        return md_df

    def get_images(self):
        if self.doctype=="pdf":
            image_names = []
            for i in range(len(self.doc)):
                
                for img in self.doc.getPageImageList(i):
                    xref = img[0]
                    pix = fitz.Pixmap(self.doc, xref)
                    image = np.array(Image.open(io.BytesIO(pix.getImageData())))
                    pixies = []
                    for abc in image:
                        for ab in abc:
                            pixies.append((ab[0], ab[1], ab[2]))
                    if len(set(pixies)) > 1:
                        if pix.n < 5:  # this is GRAY or RGB
                            pix.writePNG(os.path.join(self.file_path, "p%s-%s.png" % (i, xref)))
                            image_names.append("p%s-%s.png" % (i, xref))
                        else:  # CMYK: convert to RGB first
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                            pix.writePNG(os.path.join(self.file_path, "p%s-%s.png" % (i, xref)))
                            image_names.append("p%s-%s.png" % (i, xref))
                    pix = None
        elif self.doctype=="docx":
            image_names=[]
        
        return image_names

    def get_tables(self):
        if self.doctype=="pdf":
            tables = []
            for i in range(0, len(self.doc)):
                table = tabula.read_pdf(os.path.join(self.file_path, self.file_name), pages=i, multiple_tables=True)
                [tables.append(i) for i in table]
        elif self.doctype=="docx":
            tables=[]
            for table in self.doc.tables:
                rows=[]
                for row in table.rows:
                    cells=[]
                    for cell in row.cells:
                        cells.append(cell.text)
                    rows.append(cells)
                    df=pd.DataFrame(rows)
                    df.columns= df.iloc[0]
                    df.drop(0, inplace=True)
                tables.append(df)
                    
        return tables


#fg= read_all(file_path=r"C:\Users\tarun\Downloads", file_name="202106_App_Annie_IDC_Gaming_Spotlight_EN.pdf")
#fg.get_metadata()
"C:\Users\tarun\Downloads\What consumers want_Final.pdf"
rd=read_all(file_path=r"C:\Users\tarun\Downloads", file_name="What consumers want_Final.pdf")

rd2=read_all(file_path=r"C:\Users\tarun\Downloads", file_name="Whitepaper_What Consumers Want.docx")

rd2=read_all(file_path=r"C:\Users\tarun\Desktop", file_name="test.docx")

for p in rd2.doc.paragraphs:
    print(p.text)

jk= rd2.get_metadata()
jk= rd2.get_tables()




