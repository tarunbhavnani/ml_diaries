#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('C:\\Users\\ELECTROBOT\\Desktop')


# In[2]:


import fitz


# In[17]:


doc = fitz.open("Heroku+Tutorial.pdf")
page=doc[0]
text="heroku"
text_instances = page.searchFor(text)


for inst in text_instances:
    #highlight = page.addHighlightAnnot(inst)
    #highlight = page.addSquigglyAnnot(inst)
    #highlight = page.addUnderlineAnnot(inst)
    highlight = page.addStrikeoutAnnot(inst)
    #highlight.setColors({"stroke":(0, 0, 1), "fill":(0.75, 0.8, 0.95)})
    
    highlight.update()
doc.save("output.pdf")


# In[ ]:




