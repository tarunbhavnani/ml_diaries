from fastapi import FastAPI, Depends, File, UploadFile, Request, Form
from app.routers import *
from functions import process_uploaded_files, Qnatb, delete_files, get_final_response, get_redis_keys, allowed_file
import os
import time
import string
from typing import List, Union, Any
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import random
from pathlib import Path
import jinja2
if hasattr(jinja2, "pass_context"):
	pass_context= jinja2.pass_context
else:
	pass_context= jinja2.contextfunction


app= FastAPI(root_path= os.environ.get("SCRIPT_NAME"), dependencies= [Depends(fetch_user)])
app.include(predict_router)






from fastapi.middleware.cors import CORSMiddleware
origins=["https://127.0.0.1:8080", "https://localhost:8080"]
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
	)

BASE_DIR="./nlp-qna/app"

app.mount("/static", StaticFiles(directory=str(Path(BASE_DIR, "static"))), name="static")

deg get_templates():
return Jinja2Templates(directory= str(Path(BASE_DIR, "templates")))

templates= get_templates()
@pass_context
def https_url_for(context:dict, name:str, **path_params) -> str:
	request= context["request"]
	https_url= request.url_for(name, **path_params)
	return str(http_url).replace("http", "https")


if ENVIRONMENT!='LOCAL':
	templates.env.globals["url_for"]=https_url_for




@app.get("/", response_class= HTMLResponse, dependencies=[Depends(get_templates)])
async def index(request: Request, user=Depends(fetch_user)):
	names= get_redis_keys(collection=None, user_id= str(user.id))
	names= [i for i in names if allowed_file(i)]
	return templates.TemplateResponse("index.html", context= {"request":request, "file_names":names})

@app.post("/", response_class=HTMLResponse, dependencies=[Depends(get_templates)])
async def upload_files_web(request:Request, collection_var= None, user=Depends(fetch_user), files: List(UploadFile)= File(...)):
	_- process_uploaded_files(files= files, collection=collection_var, user_id= str(user.id))
	names= get_redis_keys(collection=None, user_id= ste(user.id))
	names= [i for i in names if allowed_file(i)]
	return templates.TemplateResponse("index.html", context= {"request":request, "file_names":names})

@app.get("/reset/web", response_class=HTMLResponse, dependencies=[Depends(get_templates)])
async def reset_web(request:Request, user= Depends(fetch_user)):
	_= delete_files(collection=None, user_id= str(user.id))
	names= get_redis_keys(collection=None, user_id= ste(user.id))
	names= [i for i in names if allowed_file(i)]
	return templates.TemplateResponse("index.html", context= {"request":request, "file_names":names})




qna=None
def load_qna():
	global qna
	if qna is None:
		qna= Qnatb(model_path=model_path)



model_path=""



@app.post("/predict/web", response_class=HTMLResponse, dependencies=[Depends(get_templates)])
async def predict_web(request:Request, search: str=Form(...), user= Depends(fetch_user)):
	load_qna()
	response_class=get_final_response(qna, question= search, collection=None, user_id=str(user.id))
	context= {"request":request, "responses": responses['results'], "search_query": search}
	return templates.TemplateResponse("search.html", context=context)









