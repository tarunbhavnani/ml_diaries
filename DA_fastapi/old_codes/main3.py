from typing import List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from functions3 import process_uploaded_files, Qnatb, delete_files,get_final_responses,get_file_names
import os
from fastapi import FastAPI, HTTPException


path= os.getcwd()

UPLOAD_FOLDER=r'C:\Users\ELECTROBOT\Desktop\git\ml_diaries\DA_fastapi\uploads'




qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')


app = FastAPI()



@app.get("/")
async def read_main():
    return {"msg": "Hello World"}



@app.post("/files")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        process_uploaded_files(files)
    except Exception as e:
        delete_files()
        raise e
    

def load_qna():
    global qna
    if qna is None:
        qna = Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')



class request_body(BaseModel):
    text: str


@app.post('/predict')
async def predict(data: request_body)-> List[dict]:
    # load file processor object
    try:
        #load_qna()
        responses = get_final_responses(qna, search_data=data.text)

        return responses
    except Exception as e:
        print(str(e))
        raise e






@app.get("/reset")
def reset():
    try:
        delete_files()
    except Exception as e:
        raise e
    

@app.get("/filename")
def filenames():
    file_names=get_file_names()
    return {"file_names":file_names}
    