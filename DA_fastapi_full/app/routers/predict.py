from fastapi import APIRouter, Depends, Request, UploadFile, File
from app.dependencies import fetch_user
from typing import Union, List
from app.functions import Qnatb, get_final_responses
from app.logger import logger
from app.response_def import RequestBody, ResponseItemPredict, ResponsePredict, ResponseItemPredict1, ListResponse
from app.evironment import ENVIRONMENT

router= APIRouter(
    prefix="/predict",
    tags= ["predict"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


#####

qna=None

def load_qna():
    global qna
    if qna is None:
        qna= Qnatb(model_path=model_path)

if ENVIRONMENT=="LOCAL":
    model_path=""
    load_qna()
else:
    model_path=""
    load_qna()


@router.post("/", response_model= ResponseItemPredict1)
def predict(data= RequestBody, user= Depends(fetch_user))->ResponsePredict:
    load_qna()
    responses= get_final_responses(qna, question= data.text, collection=data.collection, user_id= str(user.id))
    response_items=[]

    for item in responses['results']:
        response_items.append(ResponseItemPredict(
                                answer=item['answer'],
                                blob=item['blob'],
                                logits=item['logits'],
                                doc=item['docs'],
                                page=item['page'],
                                sentence=item['sentence']))
    return ResponseItemPredict1(errormsg=responses['erroormsg'], results=ResponseItemPredict(responses=response_items), status=responses['status'])






