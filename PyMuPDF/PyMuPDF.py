# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:26:32 2021

@author: ELECTROBOT
"""

# imports 
import fitz 
import re 
print(fitz.__doc__)

file=r"C:\Users\ELECTROBOT\Desktop\covid.pdf"

doc = fitz.open(file) #

doc.metadata

doc.get_toc()
doc.loadPage(page_id=1)



page = doc.loadPage(1) 
page = doc[11]#same


# for page in doc:
# # do something with 'page'
# # ... or read backwards
# for page in reversed(doc):
# # do something with 'page'
# # ... or even use 'slicing'
# for page in doc.pages(start, stop, step):
# # do something with 'page'

# # get all links on a page
# links = page.getLinks()


#save as image
pix = page.getPixmap()
pix.writeImage("page-%i.png" % page.number)



# =============================================================================
# #extract all text
# =============================================================================

import sys, fitz
#fname = sys.argv[1] # get document filename
doc = fitz.open(file) # open document
out = open(file + ".txt", "wb") # open text output
for page in doc: # iterate the document pages
    text = page.getText().encode("utf8") # get plain text (is in UTF-8)
    out.write(text) # write text of page
    out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
out.close()




#to maintain natural order
doc = fitz.open(file)
header = "Header" # text in header
footer = "Page %i of %i" # text in footer
for page in doc:
    page.insertText((50, 50), header) # insert header
    page.insertText( # insert footer 50 points above page bottom
    (50, page.rect.height - 50),
    footer % (page.number + 1, len(doc)),
)





# =============================================================================
##underline words in pdf 
# =============================================================================


def mark_word(page, text):
    """Underline each word that contains 'text'.
    """
    found = 0
    wlist = page.getTextWords() # make the word list
    for w in wlist: # scan through all words on page
        if text in w[4]: # w[4] is the word's string
            found += 1 # count
            r = fitz.Rect(w[:4]) # make rect from word bbox
            page.addUnderlineAnnot(r) # underline
    return found


fname = file

text = "Roman"#" to modern times"#sys.argv[2] # search string
doc = fitz.open(fname)
print("underlining words containing '%s' in document '%s'" % (word, doc.name))
new_doc = False # indicator if anything found at all
for page in doc: # scan through the pages
    found = mark_word(page, text) # mark the page's words
    if found: # if anything found ...
        new_doc = True
        print("found '%s' %i times on page %i" % (text, found, page.number + 1))
if new_doc:
    doc.save("marked.pdf")











# =============================================================================
# How to Extract Text in Natural Reading Order
# =============================================================================

doc = fitz.open("some.pdf")
header = "Header" # text in header
footer = "Page %i of %i" # text in footer
for page in doc:
    page.insertText((50, 50), header) # insert header
    page.insertText( # insert footer 50 points above page bottom
    (50, page.rect.height - 50),
    footer % (page.number + 1, len(doc)),
    )
# The text sequence extracted from a page modified in this way will look like this:
# 1. original text
# 2. header line
# 3. footer line

# =============================================================================
# recopver broken words
# =============================================================================
rect=pix.irect
words= page.getTextWords()
from operator import itemgetter
from itertools import groupby
def recover(words, rect):
    """ Word recovery.
    Notes:
    Method 'getTextWords()' does not try to recover words, if their single
    letters do not appear in correct lexical order. This function steps in
    here and creates a new list of recovered words.
    Args:
    words: list of words as created by 'getTextWords()'
    rect: rectangle to consider (usually the full page)
    Returns:
    List of recovered words. Same format as 'getTextWords', but left out
    block, line and word number - a list of items of the following format:
    [x0, y0, x1, y1, "word"]
    """
    # build my sublist of words contained in given rectangle
    mywords = [w for w in words if fitz.Rect(w[:4]) in rect]
    # sort the words by lower line, then by word start coordinate
    mywords.sort(key=itemgetter(3, 0)) # sort by y1, x0 of word rectangle
    # build word groups on same line
    grouped_lines = groupby(mywords, key=itemgetter(3))
    words_out = [] # we will return this
    # iterate through the grouped lines
    # for each line coordinate ("_"), the list of words is given
    for _, words_in_line in grouped_lines:
        for i, w in enumerate(words_in_line):
            if i == 0: # store first word
                x0, y0, x1, y1, word = w[:5]
                continue
            r = fitz.Rect(w[:4]) # word rect
            # Compute word distance threshold as 20% of width of 1 letter.
            # So we should be safe joining text pieces into one word if they
            # have a distance shorter than that.
            threshold = r.width / len(w[4]) / 5
            if r.x0 <= x1 + threshold: # join with previous word
                word += w[4] # add string
                x1 = r.x1 # new end-of-word coordinate
                y0 = max(y0, r.y0) # extend word rect upper bound
                continue
            # now have a new word, output previous one
            words_out.append([x0, y0, x1, y1, word])
            # store the new word
            x0, y0, x1, y1, word = w[:5]
            # output word waiting for completion
        words_out.append([x0, y0, x1, y1, word])
    return words_out


# =============================================================================
# How to Analyze Font Characteristics
# =============================================================================

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
doc = fitz.open("text-tester.pdf")
page = doc[0]
# read page text as a dictionary, suppressing extra spaces in CJK fonts
blocks = page.getText("dict", flags=11)["blocks"]

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








# =============================================================================
# redact
# =============================================================================
class Redactor: 
    
    # static methods work independent of class object 
    @staticmethod
    def get_sensitive_data(lines): 
        
        """ Function to get all the lines """
          
        # email regex 
        EMAIL_REG = r"([\w\.\d]+\@[\w\d]+\.[\w\d]+)"
        for line in lines: 
            
            # matching the regex to each line 
            if re.search(EMAIL_REG, line, re.IGNORECASE): 
                search = re.search(EMAIL_REG, line, re.IGNORECASE) 
                  
                # yields creates a generator 
                # generator is used to return 
                # values in between function iterations 
                yield search.group(1) 
  
    # constructor 
    def __init__(self, path): 
        self.path = path 
  
    def redaction(self): 
        
        """ main redactor code """
          
        # opening the pdf 
        doc = fitz.open(self.path) 
        doc = fitz.open(file) 
          
        # iterating through pages 
        for page in doc: 
            
            # _wrapContents is needed for fixing 
            # alignment issues with rect boxes in some 
            # cases where there is alignment issue 
            page._wrapContents() 
              
            # geting the rect boxes which consists the matching email regex 
            sensitive = self.get_sensitive_data(page.getText("text") .split('\n')) 
            for data in sensitive: 
                areas = page.searchFor(data) 
                  
                # drawing outline over sensitive datas 
                [page.addRedactAnnot(area, fill = (0, 0, 0)) for area in areas] 
                  
            # applying the redaction 
            page.apply_redactions() 
              
        # saving it to a new pdf 
        doc.save('redacted.pdf') 
        print("Successfully redacted") 
# path = 'testing.pdf'
# redactor = Redactor(path) 
# redactor.redaction()




