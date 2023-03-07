from typing import List

from fastapi import FastAPI, File, UploadFile, Request, Form
from pydantic import BaseModel
from functions4 import process_uploaded_files, Qnatb, delete_files,get_final_responses,get_file_names,upload_fp
import os
from fastapi import FastAPI, HTTPException
import shutil


from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

path= os.getcwd()

UPLOAD_FOLDER=r'C:\Users\ELECTROBOT\Desktop\git\ml_diaries\DA_fastapi\uploads'

#qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')
qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\Bert-qa\model')


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

########################################################################################################################

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
        load_qna()
        responses = get_final_responses(qna, question=data.text)
        return responses
    except Exception as e:
        print(str(e))
        raise e

########################################################################################################################

@app.get("/reset")
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

########################################################################################################################

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, search: str = Form(...)):
    # Implement the search functionality here
    results = get_final_responses(qna, search_data=search)
    context = {"request": request, "results": results, "search_query": search}
    return templates.TemplateResponse("search.html", context)