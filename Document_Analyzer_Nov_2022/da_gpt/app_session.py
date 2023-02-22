from flask import Flask, render_template, request, redirect,send_from_directory
from werkzeug.utils import secure_filename
import os
import _pickle as pickle
from endpoints.QnA_gpt import qnatb
import shutil


app = Flask(__name__)
qna = qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')

uploads = os.path.join(os.getcwd(), 'uploads')


def get_user_name():
    return "m554417"

UPLOAD_FOLDER= os.path.join(uploads,get_user_name())


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#if not os.path.isdir(app.config['UPLOAD_FOLDER']):
#    os.mkdir(app.config['UPLOAD_FOLDER'])

allowed_ext = [".pdf"]
qna_cached = None


def load_qna_cached():
    global qna_cached
    if qna_cached is None:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'qna'), 'rb') as handle:
            qna_cached = pickle.load(handle)
    return qna_cached

def delete_qna_cached():
    global qna_cached
    qna_cached=None
    return qna_cached

def allowed_file(file):
    return True in [file.endswith(i) for i in allowed_ext]

@app.route('/', methods=["GET", "POST"])
def index_page():
    if request.method == "POST":
        try:
            reset_files()
            
            files = request.files.getlist('files[]')
            if not os.path.isdir(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])
            
            
            for file in files:
                if file and allowed_file(file.filename):
                    try:
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    except Exception as e:
                        print(f"Error occurred while processing file: {e}")
            names = [os.path.join(app.config['UPLOAD_FOLDER'], i) for i in os.listdir(app.config['UPLOAD_FOLDER']) if i.endswith('.pdf')]
            qna.files_processor_tb(names)
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'qna'), 'wb') as handle:
                pickle.dump(qna, handle)
            #global qna_cached
            #qna_cached = None
            load_qna_cached()
        except Exception as e:
            print(f"Error occurred while processing files: {e}")
            print("No readable files")

    return render_template('index.html', names=[i for i in os.listdir(app.config['UPLOAD_FOLDER']) if i.endswith('.pdf')])


#@app.route('/delete')
#def reset_files():
#    try:
#        delete_qna_cached()
#        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
#            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#            if os.path.isfile(file_path):
#                os.unlink(file_path)
#
#    except Exception as e:
#        print(e)
#        print("No resetting")
#    return redirect('/')


@app.route('/delete')
def reset_files():
    # removes files from upload folder and then cleans the session
    try:
        delete_qna_cached()
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
        search_data = request.form.get("search")
        qna_loaded = load_qna_cached()
        #responses, answer = qna_loaded.get_response_sents(question=search_data, max_length=10)
        responses=qna.get_top_n( search_data, top=100, lm=True)
        #responses=qna.get_response_cosine(search_data)
        return render_template('search.html', responses=responses, search_data=search_data)
    except Exception as e:
        print(f"Error occurred while searching: {e}")
        return redirect('/')




@app.route('/uploads/<filename>/')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

