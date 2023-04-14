import fitz
import os
import pandas as pd
import io
from PIL import Image
import numpy as np
import tabula

class PyMuPDF_all():
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name
        self.doc = fitz.open(os.path.join(file_path, file_name))

    def get_metadata(self):
        #md = self.doc.metadata
        md_df = pd.DataFrame(self.doc.metadata.items(), columns=["Parameter", "Details"])
        return md_df

    def get_images(self):
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
        return image_names

    def get_tables(self):
        tables = []
        for i in range(0, len(self.doc)):
            table = tabula.read_pdf(os.path.join(self.file_path, self.file_name), pages=i, multiple_tables=True)
            [tables.append(i) for i in table]
        return tables