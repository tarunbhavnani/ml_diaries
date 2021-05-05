#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


os.chdir('C:\\Users\\ELECTROBOT\\Desktop')


# In[3]:


import fitz
def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)


# In[4]:


doc = fitz.open("Heroku+Tutorial.pdf")
page = doc[0]


# In[5]:


page


# In[6]:


# read page text as a dictionary, suppressing extra spaces in CJK fonts
blocks = page.get_text("dict", flags=11)["blocks"]

for b in blocks: # iterate through the text blocks
    for l in b["lines"]: # iterate through the text lines
        for s in l["spans"]: # iterate through the text spans
            print("")
            font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                s["font"], # font name
                flags_decomposer(s["flags"]), # readable font flags
                s["size"], # font size
                s["color"], # font color
            )
            print("Text: '%s'" % s["text"]) # simple print of text
            print(font_properties)


# In[ ]:




