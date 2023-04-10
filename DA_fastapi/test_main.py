from fastapi.testclient import TestClient

from main import app
from functions import process_uploaded_files, Qnatb, delete_files,get_final_responses,get_file_names,upload_fp

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

import os
import pickle
import pytest

from functions import get_final_responses
from unittest.mock import Mock, mock_open, patch


import pytest
from unittest.mock import patch, MagicMock
from functions import Filetb

import os
import pytest
from unittest.mock import patch, Mock
from functions import Filetb

@pytest.fixture
def filetb():
    return Filetb()

def test_clean():
    assert Filetb.clean("<p>This is a test</p>") == ' This is a test '
#    assert Filetb.clean("<a href='http://example.com'>Example</a>") == "Example"
#    assert Filetb.clean("The quick brown fox jumps over the lazy dog.") == "The quick brown fox jumps over the lazy dog."
#    assert Filetb.clean("For internal use only 1 of 2") == ""

def test_split_into_sentences():
    text = "This is a test. This is only a test! Mr. Smith went to Washington."
    sentences = Filetb.split_into_sentences(text)
    assert sentences == ['This is only a test!']

def test_files_processor_tb(filetb):
    files = ['file1.pdf', 'file2.pdf']
    with patch('functions.fitz.open') as mock_open:
        mock_page = Mock()
        mock_page.get_text.return_value = "This is a test. This is only a test! Mr. Smith went to Washington."
        mock_doc = Mock()
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value = mock_doc
        tb_index, all_sents, vec, tfidf_matrix = filetb.files_processor_tb(files)
        assert len(tb_index) == 6
        assert len(all_sents) == 6
        assert vec is not None
        assert tfidf_matrix is not None

def test_get_response_cosine(filetb):
    filetb.tb_index = [
        {'doc': 'file1.pdf', 'page': 0, 'sentence': 'This is a test.'},
        {'doc': 'file1.pdf', 'page': 0, 'sentence': 'This is only a test!'},
        {'doc': 'file2.pdf', 'page': 0, 'sentence': 'Mr. Smith went to Washington.'},
    ]
    filetb.all_sents = ['this is a test', 'this is only a test', 'mr. smith went to washington']
    filetb.vec = Mock()
    filetb.vec.transform.return_value = [[1, 0, 0]]
    with patch.object(filetb, 'tfidf_matrix', [[0.1], [0.2], [0.3]]):
        response = filetb.get_response_cosine("What is this?")
        assert len(response) == 3
    response = filetb.get_response_cosine("What is this?")
    assert len(response) == 2
    assert response[0]['doc'] == 'file1.pdf'
    assert response[0]['page'] == 0
    assert response[0]['sentence'] == 'This is only a test!'
    assert response[1]['doc'] == 'file1.pdf'
    assert response[1]['page'] == 0
    assert response[1]['sentence'] == 'This is a test.'