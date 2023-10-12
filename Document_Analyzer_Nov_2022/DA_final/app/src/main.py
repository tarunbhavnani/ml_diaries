from flask import Flask, render_template, request
import os

from . import app
from .endpoints.functions import Qnatb, get_file_names, process_uploaded_files, delete_files, load_fp, send_file, get_final_responses

qna = Qnatb(model_path=r'D:\model_dump\minilm-uncased-squad2')

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])


@app.route('/', methods=["GET", "POST"])
def index_page():
    
    file_names= get_file_names()
    
    if request.method == "POST":
        try:
            
            files = request.files.getlist('files[]')
        
            process_uploaded_files(files)
            
            file_names= get_file_names()
            
                    
        except Exception as e:
            print(f"Error occurred while processing files: {e}")
            print("No readable files")
            file_names= get_file_names()

    return render_template('index.html', names=file_names)



@app.route('/delete')
def reset_files():
    file_names= get_file_names()
    try:
        delete_files()
        file_names= get_file_names()

    except Exception as e:
        print(e)
        print("No resetting")
        file_names= get_file_names()

    return render_template('index.html', names=file_names)




@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data = request.form.get("search")

        responses=get_final_responses(qna, search_data)
        
        return render_template('search.html', responses=responses, search_data=search_data)
    
    except Exception as e:
        
        print(f"Error occurred while searching: {e}")
        file_names= get_file_names()
        
        return render_template('index.html', names=file_names)



@app.route('/uploads/<filename>/')
def upload(filename):
    return send_file(filename)







