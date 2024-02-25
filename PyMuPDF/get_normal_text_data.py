# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:27:55 2024

@author: tarun
"""


import fitz
from operator import itemgetter

def fonts(doc, granularity=False):
    """Extracts fonts and their usage in PDF documents.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles



#file=r"C:\Users\ELECTROBOT\Desktop\output.pdf"
file= r"C:/Users/tarun/Desktop/Books/AAAMLP.pdf"
doc = fitz.open(file) #
font_counts, styles = fonts(doc, granularity=False)


normal_font= float(font_counts[0][0])
#remove all with fonts lower than normal

fonts= [i for i in font_counts if i[0]>=normal_font]




all_text=[]
for page in doc:

    blocks = page.get_text("dict")["blocks"]

    for b in blocks:  # iterate through the text blocks

        if b['type'] == 0:  # block contains text
            for l in b["lines"]:  # iterate through the text lines
                for s in l["spans"]:  # iterate through the text spans
                    if s['size']>=normal_font:
                        all_text.append(s['text'])
                











