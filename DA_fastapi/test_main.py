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


def test_get_final_responses():
    # Create mock inputs
    qna = None
    question = "some_question"
    collection = "some_collection"

    # Create a mock pickle data to load
    fp = {"some_result"}

    # Mock the 'open' function to simulate reading the pickle file

    with patch('builtins.open', mock_open(read_data=pickle.dumps(fp))):
        # Call the function
        result = get_final_responses(qna, question, collection)

    # Assert the expected output
    assert result == {"some_result"}

@pytest.fixture()
def file_processor():
    return True


class Filetb:

    @pytest.fixture(autouse=True)
    def _file_processor(self, file_processor):
        self._response = file_processor

    def files_processor_tb(self):
        assert self._response == True
