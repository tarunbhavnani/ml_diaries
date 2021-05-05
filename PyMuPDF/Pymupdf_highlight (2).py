import fitz

### READ IN PDF

doc = fitz.open("input.pdf")
page = doc[0]

### SEARCH

text = "Sample text"
text_instances = page.searchFor(text)

### HIGHLIGHT

for inst in text_instances:
    highlight = page.addHighlightAnnot(inst)


### OUTPUT

doc.save("output.pdf", garbage=4, deflate=True, clean=True)