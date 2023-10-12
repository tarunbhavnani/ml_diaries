import pytest
from fastapi.testclient import TestClient
from typing import Union
from app.routers.predict import load_qna, Qnatb
from app.functions import allowed_file,Filetb, process_upload_files, get_final_responses

import io
import pickle
import torch
from unittest.mock import Mock, mock_open, patch, MagicMock

from app.main import app
from app.cache.redis import cache


client= TestClient(app)

async def fetch_user_dummy():
    return CSUser()

@pytest.fixture
def cs_user():
    return CSUser()




############

def test_allowed_file():
    assert allowed_file('abc.pdf')==True
    assert allowed_file('abc.doc')==False


@pytest.fixture()
def filetb(monkeypatch, mocker):
    monkeypatch.setattr("app.functions.Filetb.files_processor_tb", lambda files: "result")
    monkeypatch.setattr("app.functions.Filetb.get_response_cosine", lambda question: "result")
    fp= Filetb()
    return fp


def test_filetb_init(filetb):
    assert filetb.files==None
    assert filetb.index==None
    assert filetb.all_sents==None
    assert filetb.vsc==None
    assert filetb.tfidf_matrix==None
    assert filetb.stopwords is not None


def test_filetb_stem(filetb):
    assert filetb.stem("this")=="thi"

def test_filetb_clean(filetb):
    assert filetb.clean("this")=="this"

def test_filetb_split(filetb):
    assert filetb.split_into_sentences("this is a sentence")==["this is a sentence"]

##############


@pytest.fixture
def get_cache_read_response():
    return "obj"

class MockResponse:
    @staticmethod
    def get_text():
        return "unit test for fp"    

def test_files_processor_tb(monkeypatch, filetb, get_cache_read_response):
    def mock_get(*args, **kwargs):
        return [MockResponse()]
    monkeypatch.setattr("app.functions.cache.read_from_cache_user", lambda file:get_cache_read_response)
    monkeypatch.setattr("app.functions.fitz.open", lambda stream, filetype: mock_get())

    mock_file=[r"dummy_file\dummy_file"]
    filetb.files_processor_tb()
    assert filetb.tb_index is None
    assert filetb.all_sents is None
    assert filetb.vec is None
    assert filetb.tfidf_matrrix is None




def test_filetb_get_response_cosine(filetb, monkeypatch):
    resp= filetb.get_response_cosine()
    assert resp=="result"



def test_get_redis_keys(mocker, monkeypatch):
    monkeypatch.setattr("app.cache.redis.RedisCache.read_from_cache_user", lambda x:{})
    monkeypatch.setattr("app.cache.redis.RedisCache.keys", lambda x:[])
    from app.functions import get_redis_keys
    assert get_redis_keys('tarun', "kl")==[]


def test_delete_files(monkeypatch, mocker):
    monkeypatch.setattr("app.cache.redis.RedisCache.read_from_cache_user", lambda x:{})
    monkeypatch.setattr("app.cache.redis.RedisCache.keys", lambda x:[])
    from app.functions import delete_files
    assert delete_files("tarun","kl")==[]





##############



@pytest.fixture
def get_collection_response():
    return "collection_var"

@pytest.fixture
def get_pickjleload_response(filetb):
    filetb.files_processor_tb()
    return filetb

@pytest.fixture
def get_cache_response():
    return


def test_process_uploaded_files(monkeypatch, get_collection_response, get_cache_response):
    monkeypatch.setattr("app.functions.get_colection", lambda collection, user_id: get_collection_response )
    monkeypatch.setattr("app.functions.cache.write_to_cache_user", lambda filepath, obj: get_cache_response )
    monkeypatch.setattr("app.functions.pickle.dumps", lambda obj: get_cache_response)
    monkeypatch.setattr("app.cache.redis.RedisCache.read_from_cache_user", lambda x:{})
    monkeypatch.setattr("app.cache.redis.RedisCache.keys", lambda x:[])

    mock_file= MagicMock()
    mock_file.file=io.BytesIO(b"kghji")
    mock_file.filename="mock.pdf"
    result= process_upload_files([mock_file], "collection","user_id")
    assert result is not None
    
##############

#details below from model and tokenizer
class tokenizer_response:
    @staticmethod
    def encode_plus(*args, **kwargs):
        return {"input_ids":[],
                "token_type_ids":[],
                "attention_mask":[]}
    @staticmethod
    def convert_ids_to_tokens(*args, **kwargs):
        return []



@pytest.fixture
def qnatb(monkeypatch, mocker):
    model_response=Mock()
    
    model_response.return_value={"start_logits":torch.tensor([[]]),
    
                                 "end_logits":torch.tensor([[]])}
    
    monkeypatch.setattr("transformers.AutoModelForQuestionAnswering.from_prertrained" lambda x: model_response)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda x: tokenizer_response())

    from app.functions import Qnatb
    qnatb= Qnatb("model_path")
    return qnatb


def test_qnatb_init(monkeypatch, qnatb):
    assert qnatb.model_path=="model_path"
    assert qnatb.model is not None
    assert qnatb.tokenizer is not None


@pytest.fixture
def get_qna_response():
    return "result"

class pickle_response:
    @staticmethod
    def get_response_cosine(*args, **kwargs):
        return "result"

def test_get_final_responses(monkeypatch, filetb, qnatb, get_collection_response, get_cache_response, get_pickleload_response, get_qna_response):
    monkeypatch.setattr("app.functions.get_colection", lambda collection, user_id: get_collection_response )
    #monkeypatch.setattr("app.functions.cache.write_to_cache_user", lambda filepath, obj: get_cache_response )
    monkeypatch.setattr("app.functions.pickle.loads", lambda obj: pickle_response())
    monkeypatch.setattr("app.cache.redis.RedisCache.read_from_cache_user", lambda x:{})
    monkeypatch.setattr("app.cache.redis.RedisCache.keys", lambda x:[])
    monkeypatch.setattr("app.functions.Qnatb.extract_answer_blobs", lambda question, responses, num, model: [{'doc':'None', 'page':0, "sentence":'None', 'answer':'None','logits':0, 'blob':'None'}])

    result = get_final_responses(qnatb, "question", "collection", "user_id")
    assert result is not None





def test_answer_question(qnatb):
    mock_question= "who is federer"
    mock_answer_text="federer is a tennis player"

    answer, start_logit= qnatb.answer_question(mock_question, mock_answer_text)

    assert answer is not None
    assert start_logit is not None


def text_extract_answer_blobs(monkeypatch, mocker):
    monkeypatch.setattr("app.functions.Qnatb.answer_question", lambda question, answer_text, model:("roger_federer", 55,7695))
    mock_question= "who is federer"
    mock_response=[{...}]#details


    







