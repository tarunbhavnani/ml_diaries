#!/usr/bin/env python
# coding: utf-8



import fitz
import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\check')

path=r'C:\Users\ELECTROBOT\Desktop\check'

uploaded_file.save(os.path.join(path, filename))



for i in range(len(doc)):
    image_names=[]
    for img in doc.getPageImageList(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        image = np.array(Image.open(io.BytesIO(pix.getImageData()))) 
        #print('PIL:\n', image)
        #imgplot = plt.imshow(image)

        pixies=[]
        for img in image:
            for im in img:
                pixies.append((im[0],im[1], im[2]))
        if len(set(pixies))>1:

        

            if pix.n < 5:       # this is GRAY or RGB
                pix.writePNG("p%s-%s.png" % (i, xref))
                image_names.append("p%s-%s.png" % (i, xref))
                
                
            else:               # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.writePNG("p%s-%s.png" % (i, xref))
                image_names.append("p%s-%s.png" % (i, xref))
        pix=None
        
        
        
        
def get_images(doc, path):
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









# your Pillow code
import io
from PIL import Image
import numpy as np
image = np.array(Image.open(io.BytesIO(pix.getImageData()))) 
print('PIL:\n', image)
imgplot = plt.imshow(image)


# import os
# os.getcwd()

# img=pix.getPNGData()
# img=pix.toImage()

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# img=mpimg.imread(r"C:\Users\ELECTROBOT\Pictures\tarun_professional.jpg")

# imgplot = plt.imshow(img)

pixies=[]
for img in image:
    for im in img:
        pixies.append((im[0],im[1], im[2]))
len(set(pixies))==1












