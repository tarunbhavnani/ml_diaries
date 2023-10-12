import pytest
from fastapi.testclient import TestClient
from typing import Union
from fastapi import APIRouter, Depends, Request, UploadFile, File

from unittest import mock
from app.main import app

client= TestClient(app)

async def fetch_user_dummy():
    return CSUser()


app.dependency_overrides[fetch_user]=fetch_user_dummy

@pytest.fixture
def cs_user():
    return CSUser()

@pytest.fixture
def reset_response()
    return ['file1','file2']


@pytest.fixture
def headers():
    return {}



def test_reset(monkeypatch, reset_response, cs_user):
    monkeypatch.setattr("app.authentication.authenticate_user", lambda x,y:cs_user)
    monkeypatch.setattr("app.routers.reset.delete_files", lambda collection, user_id: reset_response)
    
    response= client.delete('/reset/{collection_var}')

    assert response.status_code==200
    assert response.json == {"names":['file1','file2']}

    
