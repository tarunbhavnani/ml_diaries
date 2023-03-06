from typing import List, Union

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from functions import process_uploaded_files, Qnatb, delete_files,get_final_responses,get_file_names,upload_fp
import os
from fastapi import FastAPI, HTTPException
import shutil

path= os.getcwd()

UPLOAD_FOLDER=r'C:\Users\ELECTROBOT\Desktop\git\ml_diaries\DA_fastapi\uploads'

#qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')
qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\Bert-qa\model')


app = FastAPI()


########################################################################################################################
@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

########################################################################################################################

@app.post("/uploadfiles")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        process_uploaded_files(files)
    except Exception as e:
        delete_files()
        raise e

@app.post("/upload_folder/")
async def upload_folder(folder: UploadFile = File(...)):
    # process the uploaded folder
    # here you can write code to save the uploaded folder to disk or process it in memory
    return {"message": "Folder uploaded successfully"}


@app.post("/uploadfiles_collection/<collection_var>")
async def upload_files(collection_var,files: List[UploadFile] = File(...)):
    try:
        process_uploaded_files(files, collection=collection_var)
    except Exception as e:
        delete_files()
        raise e

########################################################################################################################

def load_qna():
    global qna
    if qna is None:
        qna = Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')

class request_body(BaseModel):
    text: str
    collection: Union[str, None]=None

class ResponseItem(BaseModel):
    doc: str
    page: int
    sentence: str
    answer: str
    logits: float
    blob: str

class Response(BaseModel):
    responses: List[ResponseItem]

@app.post('/predict')
async def predict(data: request_body)-> Response:
    # load file processor object
    try:
        load_qna()
        responses = get_final_responses(qna, question=data.text, collection=data.collection)
        return responses
    except Exception as e:
        print(str(e))
        raise e

########################################################################################################################

@app.delete("/reset")
def reset():
    try:
        delete_files()
    except Exception as e:
        raise e

########################################################################################################################

@app.get("/filename")
def file_name():
    file_names = get_file_names()
    return {"file_names": file_names}

########################################################################################################################

@app.post("/uploadobject/")
async def upload_object(file: UploadFile = File(...)):
    delete_files()
    upload_fp(file)
    return {"filename": file.filename}