from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify, Response
#from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import fitz
import pickle
import pandas as pd
import shutil
from endpoints.functions import load_model, get_weighted_summary_pdf, summarize


from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

model_path=r'C:\Users\tarun\Desktop\summarization_bart\model_files'
model, tokenizer= load_model(model_path)

# from QnA_full_class_no_lm import qnatb

app = Flask(__name__)
app.config["SECRET_KEY"] = "OCML3BRawWEUeaxcuKHLpw"
# to use sesison secret key is needed

# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Get current path
path = os.getcwd()

# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

allowed_ext = [".pdf", ".pptx", ".docx"]


def allowed_file(file):
    return True in [file.endswith(i) for i in allowed_ext]


@app.route('/')
def index_page():
    try:
        s = os.listdir(session['Folder'])
        names=[i for i in s if i.split('.')[-1] in ['pdf', 'pptx', 'docx']]
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
            session['audit_check']= request.form.getlist('audit_report')
            print(files)
            print(session['audit_check'])
        except Exception as e:
            print(e)
            #app.logger.error(e)

        session['uid'] = uuid.uuid4().hex

        Folder = os.path.join(app.config['UPLOAD_FOLDER'], session['uid'])
        if not os.path.isdir(Folder):
            os.mkdir(Folder)

        session['Folder'] = Folder
        session['audit_trail']={}

        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(session['Folder'], filename))


                except Exception as e:
                    print(e)
                    print('not saved')
                    #app.logger.error(e)
        try:
            names = [os.path.join(session['Folder'], i) for i in os.listdir(session['Folder'])]
            # glob.glob()


        except Exception as e:
            print(e)
            #app.logger.error(e)
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
        shutil.rmtree(session['Folder'])
        session.pop('Folder', None)
    except Exception as e:
        print(e)
        print("No resetting")

    return redirect('/')





@app.route('/uploads/<filename>/')
def upload(filename):
    return send_from_directory(session['Folder'], filename)


@app.route('/summary/<filename>/')
def summary(filename):
    doc=fitz.open(os.path.join(session['Folder'], filename))
    read=""
    for page in doc:
        read+=page.getText()
    weighted_summary= get_weighted_summary_pdf(doc)
    summary= summarize(weighted_summary, tokenizer, model)
    

    return render_template("summary.html",filename=filename, read=read, weighted_summary=weighted_summary, summary=summary, len=len)



