# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:08:24 2021

@author: ELECTROBOT
pip install tabula-py
"""

import tabula
import fitz
import pandas as pd

doc=fitz.open(r"C:\Users\ELECTROBOT\Desktop\Desktop_kachra\marked.pdf")
print(len(doc))

tables=[]
for i in range(0,len(doc)):
    print(i)
    table=tabula.read_pdf(r"C:\Users\ELECTROBOT\Desktop\Desktop_kachra\marked.pdf", pages=i, multiple_tables=True)
    [tables.append(i) for i in table]


doc = fitz.open(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
md = doc.metadata
df = pd.DataFrame(md.items(), columns=["Parameter", "Details"])
tables = [df.to_html(classes='data')]

def get_images_and_tables(file_path, path):
    
    doc= fitz.open(file_path)
    tables=[pd.DataFrame(doc.metadata.items(), columns=["Parameter", "Details"]).to_html(classes='data')]
    
    image_names=[]

    for i in range(len(doc)):
        for img in doc.getPageImageList(i):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            image = np.array(Image.open(io.BytesIO(pix.getImageData())))
            pixies = []
            for img in image:
                for im in img:
                    pixies.append((im[0], im[1], im[2]))
            if len(set(pixies)) > 2:
                if pix.n < 5:  # this is GRAY or RGB
                    pix.writePNG(os.path.join(path, "p%s-%s.png" % (i, xref)))
                    image_names.append("p%s-%s.png" % (i, xref))
                else:  # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.writePNG(os.path.join(path, "p%s-%s.png" % (i, xref)))
                    image_names.append("p%s-%s.png" % (i, xref))
            pix = None
        
        table=tabula.read_pdf(r"C:\Users\ELECTROBOT\Desktop\Desktop_kachra\marked.pdf", pages=i, multiple_tables=True)
        [tables.append(i.to_html(classes='data')) for i in table]
    
    return image_names, tables


