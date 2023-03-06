import os
import shutil
import pytest
from mock import Mock
from my_module import process_uploaded_files

@pytest.fixture
def mock_upload_folder(tmpdir):
    return tmpdir.mkdir('uploads')

@pytest.fixture
def mock_files(mock_upload_folder):
    file1 = Mock(filename='file1.pdf', file=Mock(read=lambda: b'some file content'))
    file2 = Mock(filename='file2.txt', file=Mock(read=lambda: b'some file content'))
    file3 = Mock(filename='file3.pdf', file=Mock(read=lambda: b'some file content'))
    return [file1, file2, file3]

def test_process_uploaded_files(mock_files, mock_upload_folder):
    result = process_uploaded_files(mock_files, 'test_collection', upload_folder=str(mock_upload_folder))
    assert result is None
    user_folder = os.path.join(str(mock_upload_folder), 'test_collection')
    assert os.path.isdir(user_folder)
    file_path1 = os.path.join(user_folder, 'file1.pdf')
    assert os.path.exists(file_path1)
    file_path2 = os.path.join(user_folder, 'file2.txt')
    assert not os.path.exists(file_path2)
    file_path3 = os.path.join(user_folder, 'file3.pdf')
    assert os.path.exists(file_path3)


