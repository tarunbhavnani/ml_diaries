from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.functions import reg_ind
from typing import List, Union

router = APIRouter(
    prefix="/search",
    tags=["search"],
    dependencies=[],
    responses={404: {"description": "Not Found"}},
)

class ResponseItem(BaseModel):
    doc: str
    page: int
    sentence: str

class ResponsePredict(BaseModel):
    responses: List[ResponseItem]

class RequestBody(BaseModel):
    text: str
    

@router.post("/", response_model=ResponsePredict)
def search(data: RequestBody) -> ResponsePredict:
    tb_index_reg, overall_dict, docs = reg_ind(words=data.text, upload_dir="uploads")

    response_items = [
        ResponseItem(doc=item['doc'], page=item['page'], sentence=item['sentence'])
        for item in tb_index_reg
    ]

    return ResponsePredict(responses=response_items)
