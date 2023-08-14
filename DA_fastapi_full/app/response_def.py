from pydantic import BaseModel
from typing import List, Union

class ListResponse(BaseModel):
	names:List(str)


class RequestBody(BaseModel):
	text:str
	collection: Union[str,None]=None

class ResponseItemPredict(BaseModel):
	doc:str
	page:int
	sentence:str
	answer:str
	logits:float
	blob:str

class ResponsePredict(BaseModel):
	responses: List[ResponseItemPredict]


class ResponseItemPredict1(BaseModel):
	errormsg:str
	results:ResponsePredict
	status:str

	