#!/usr/bin/env python
# coding: utf-8

# In[4]:


import fitz
import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\check')


# In[5]:


doc = fitz.open(r"C:\Users\ELECTROBOT\Desktop\Arria1.pdf")
for i in range(len(doc)):
    for img in doc.getPageImageList(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:       # this is GRAY or RGB
            pix.writePNG("p%s-%s.png" % (i, xref))
        else:               # CMYK: convert to RGB first
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.writePNG("p%s-%s.png" % (i, xref))
            pix1 = None
        pix = None


# In[ ]:




