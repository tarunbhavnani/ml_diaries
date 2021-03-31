#!/usr/bin/env python
# coding: utf-8

# In[1]:


#tarun37
import speech_recognition as sr


# In[2]:


r = sr.Recognizer()


# In[11]:


harvard = sr.AudioFile(r'C:\Users\ELECTROBOT\Desktop\speach_text\output.wav')


# In[12]:


with harvard as source:
  audio = r.record(source)


# In[13]:


r.recognize_google(audio)


# In[ ]:




