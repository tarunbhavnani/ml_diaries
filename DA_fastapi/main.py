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
    file_paths = await save_files(files)
    try:
        fp = await process_files(file_paths)
        await save_file(fp)
    except Exception as e:
        file_paths.append(str(e))
        await delete_files(file_paths)
        raise e

    return {"file_name": "Good"}

async def save_files(files):
    file_paths = []
    for file in files:
        file_path = f"./uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
    return file_paths

async def process_files(file_paths):
    return file_processor(file_paths)

async def save_file(fp):
    with open(f"./uploads/fpp", "wb") as buffer:
        pickle.dump(fp, buffer)

async def delete_files(file_paths):
    for file_path in file_paths:
        os.remove(file_path)


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
