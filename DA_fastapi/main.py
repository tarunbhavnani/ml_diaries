from typing import List
import shutil
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from functions import *
import os, glob
#from pathlib import *
#current_dir = Path.cwd()
import os
import pickle
path= os.getcwd()
upload_path= os.path.join(path, "uploads")
from fastapi.responses import HTMLResponse

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
qna= qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\Bert-qa\model')

app = FastAPI()


# @app.post("/")
# async def root(file: UploadFile = File(...)):
#    with open(f"./uploads/{file.filename}", "wb") as buffer:
#        shutil.copyfileobj(file.file, buffer)
#    return {"file_name": file.filename}

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}



@app.post("/files")
async def upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        with open(f"./uploads/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        files= [os.path.join(upload_path, i) for i in os.listdir("./uploads")]
        try:
            fp = file_processor(files)
            #with open(f"./uploads/fp", "wb") as buffer:
            #    shutil.copyfileobj(fp, buffer)
            with open(f"./uploads/fpp", "wb") as buffer:
                pickle.dump(fp, buffer)

        except Exception as e:
            files.append(str(e))
    return {"file_name": "Good"}


class request_body(BaseModel):
    text: str


@app.post('/predict')
def predict(data: request_body):
    #load file processor object
    try:
        with open(f"./uploads/fpp", 'rb') as handle:
            fp = pickle.load(handle)
        # Predicting the Class
        final_response_dict = get_response_fuzz(question=data.text,
                                                vec=fp.vec,
                                                tfidf_matrix=fp.tfidf_matrix,
                                                tb_index=fp.tb_index,
                                                stopwords=fp.stopwords,
                                                max_length=7)
        LM_final = qna.get_top_n(data.text, final_response_dict, top=10, max_length=None)



        # Return the Result
        if len(LM_final)>0:
            return {'class': LM_final[0]["answer"]}
        else:
            return {'class': "No Results"}
    except Exception as e:
        return {'class': "No files"}
        #return {'class': str(e)}




@app.get("/reset")
def reset():
    for file in os.listdir("./uploads"):
        print(file)
        try:
            path="./uploads/"+ file
            os.remove(path)
        except Exception as e:
            print(e)
    return {"file_name": "Good"}
