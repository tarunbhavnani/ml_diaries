from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify, Response

from werkzeug.utils import secure_filename
import os

import _pickle as pickle
from endpoints.QnA_no_lm import qnatb
import shutil

app = Flask(__name__)
qna = qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\Bert-qa\model')
# Get current path
path = os.getcwd()
print(path)

# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

allowed_ext = [".pdf"]
qna_cached = None

def load_qna_cached():
    global qna_cached
    if qna_cached is None:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'qna'), 'rb') as handle:
            qna_cached = pickle.load(handle)
    return qna_cached

def allowed_file(file):
    return True in [file.endswith(i) for i in allowed_ext]


@app.route('/')
def index_page():
    try:
        s = os.listdir(app.config['UPLOAD_FOLDER'])
        names=[i for i in s if i.split('.')[-1] in ['pdf']]
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
            print(f"Error occurred while retrieving files: {e}")
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
                    print(f"Error occurred while saving files: {e}")
                    #app.logger.error(e)
        try:
            names = [os.path.join(app.config['UPLOAD_FOLDER'], i) for i in os.listdir(app.config['UPLOAD_FOLDER'])]
            qna.files_processor_tb(names)
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'qna'), 'wb') as handle:
                pickle.dump(qna, handle)
            qna_cached = None  # reset the cache
            _ = load_qna_cached()
        except Exception as e:
            print(f"Error occurred while processing files: {e}")
            print("No readable files")
            # app.logger.error(e)

    return redirect('/')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/delete')
def reset_files():
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.mkdir(app.config['UPLOAD_FOLDER'])
    except Exception as e:
        print(e)
        print("No resetting")
    return redirect('/')

@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data = request.form.get("search")
        qna_loaded = load_qna_cached()

        responses, answer = qna_loaded.get_response_sents(question=search_data, max_length=10)
        print(answer)
        return render_template('search.html', responses=responses, search_data= search_data, answer=answer)
    except Exception as e:
        print(e)
        return redirect('/')



@app.route('/uploads/<filename>/')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/redirect_to_index')
def redirect_to_index():
    return redirect(url_for('index_page', _external=True, scheme='https'))


