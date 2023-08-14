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


from app.logger import logger
from .environment import ENVIRONMENT
from app.dependencies import fetch_user

from app.routers import redis, predict, uploadfiles, reset, filename_redis, transferfiles, collections_available, get_pdf


app= FastAPI(root_path= os.environ.get("SCRIPT_NAME"), dependencies= [Depends(fetch_user)])
app.include_router(redis.router)
app.include_router(predict.router)
app.include_router(uploadfiles.router)
app.include_router(reset.router)
app.include_router(filename_redis.router)
app.include_router(transferfiles.router)
app.include_router(collections_available.router)
app.include_router(get_pdf.router)





from fastapi.middleware.cors import CORSMiddleware

origins=["https://127.0.0.1:8080", "https://localhost:8080"]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
	)

ENVIRONMENT='LOCAL'

if ENVIRONMENT=='LOCAL':
	BASE_DIR="./nlp-qna/app"
else:
	BASE_DIR=""

app.mount("/static", StaticFiles(directory=str(Path(BASE_DIR, "static"))), name="static")

def get_templates():
	return Jinja2Templates(directory= str(Path(BASE_DIR, "templates")))

templates= get_templates()
@pass_context
def https_url_for(context:dict, name:str, **path_params) -> str:
	request= context["request"]
	http_url= request.url_for(name, **path_params)
	return str(http_url).replace("http", "https",1)


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
	names= get_redis_keys(collection=None, user_id= str(user.id))
	names= [i for i in names if allowed_file(i)]
	return templates.TemplateResponse("index.html", context= {"request":request, "file_names":names})

@app.get("/reset/web", response_class=HTMLResponse, dependencies=[Depends(get_templates)])
async def reset_web(request:Request, user= Depends(fetch_user)):
	_= delete_files(collection=None, user_id= str(user.id))
	names= get_redis_keys(collection=None, user_id= str(user.id))
	names= [i for i in names if allowed_file(i)]
	return templates.TemplateResponse("index.html", context= {"request":request, "file_names":names})



qna=None
def load_qna():
	global qna
	if qna is None:
		qna= Qnatb(model_path=model_path)



if ENVIRONMENT=='LOCAL':
	model_path=r"D:\model_dump\Bert-qa\model"
else:
	model_path=r"D:\model_dump\Bert-qa\model"



@app.post("/predict/web", response_class=HTMLResponse, dependencies=[Depends(get_templates)])
async def predict_web(request:Request, search: str=Form(...), user= Depends(fetch_user)):
	load_qna()
	responses=get_final_response(qna, question= search, collection=None, user_id=str(user.id))
	context= {"request":request, "responses": responses['results'], "search_query": search}
	return templates.TemplateResponse("search.html", context=context)


################################################

@app.middleware("http")
async def llog_requests(request:Request, call_next):
	idem="".join(random.choices(string.ascii_uppercase+string.digits,k=6))
	logger.info(f"rid={idem} start request path={request.url.path}")
	start_time= time.time

	response= await call_next(request)

	process_time=(time.time() -start_time)*1000
	formatted_process_time='{0:2f}'.forrmat(process_time)
	logger.info(f'rrid={idem} completed_in={formatted_process_time}ms status_code={response.status_code}')
	return response


	










