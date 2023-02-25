from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify, Response
#from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import _pickle as pickle
from endpoints.QnA_no_lm_fuzz_cosine import qnatb

from endpoints.functions import PyMuPDF_all, doc_all

import pandas as pd
import shutil
import json

# from QnA_full_class_no_lm import qnatb

app = Flask(__name__)
app.config["SECRET_KEY"] = "OCML3BRawWEUeaxcuKHLpw"
# to use sesison secret key is needed


qna = qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\Bert-qa\model')
#qna= qnatb()


# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Get current path
path = os.getcwd()
print(path)

# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

allowed_ext = [".pdf"]


def allowed_file(file):
    return True in [file.endswith(i) for i in allowed_ext]


@app.route('/')
def index_page():
    try:
        s = os.listdir(app.config['UPLOAD_FOLDER'])
        names=[i for i in s if i.split('.')[-1] in ['pdf']]
        #names = [i for i in s if i.endswith(".png") != True]
        return render_template('index.html', names=names)
    except:
        return render_template('index.html')


@app.route('/', methods=["POST"])
def get_files():
    if request.method == "POST":
        try:
            reset_files()  # cleans the files from upload folder if new files uploaded and pops files from session
            files = request.files.getlist('files[]')
            print(files)
        except Exception as e:
            print(e)
            #app.logger.error(e)



        Folder = app.config['UPLOAD_FOLDER']
        if not os.path.isdir(Folder):
            os.mkdir(Folder)

        app.config['UPLOAD_FOLDER'] = Folder


        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


                except Exception as e:
                    print(e)
                    print('not saved')
                    #app.logger.error(e)
        try:
            names = [os.path.join(app.config['UPLOAD_FOLDER'], i) for i in os.listdir(app.config['UPLOAD_FOLDER'])]
            qna.files_processor_tb(names)
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'qna'), 'wb') as handle:
                pickle.dump(qna, handle)

        except Exception as e:
            print(e)
            print("No readable files")

    return redirect('/')
    #return redirect(url_for('index_page'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/delete')
def reset_files():
    # removes files from upload folder and then cleans the session
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        #session.pop('Folder', None)
        os.mkdir(app.config['UPLOAD_FOLDER'])
    except Exception as e:
        print(e)
        print("No resetting")

    return redirect('/')




@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data= request.form.get("search")
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'qna'), 'rb') as handle:
            qna_loaded= pickle.load(handle)

        responses= qna_loaded.get_top_n(search_data, top=5, max_length=10, lm=True)
        return render_template('search.html', responses=responses, search_data= search_data)
    except Exception as e:
        print(e)
        return redirect('/')




@app.route('/uploads/<filename>/')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)





