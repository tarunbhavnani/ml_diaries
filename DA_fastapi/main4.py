from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import List
from functions3 import process_uploaded_files, Qnatb, delete_files,get_final_responses,get_file_names
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


qna= Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_main(request: Request):
    file_names = get_file_names()
    context = {"msg": "Hello World", "request": request, "names": file_names}
    return templates.TemplateResponse("index.html", context=context)


@app.post("/files")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        process_uploaded_files(files)
    except Exception as e:
        delete_files()
        raise e

    # Update the list of file names
    #file_names = get_file_names()

    # Redirect to the main page with the updated list of file names
    return RedirectResponse(url='/', status_code=303, headers={'Location': '/'})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, search: str = Form(...)):
    # Implement the search functionality here
    results = get_final_responses(qna, search_data=search)
    context = {"request": request, "results": results, "search_query": search}
    return templates.TemplateResponse("search.html", context)




