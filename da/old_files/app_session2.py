from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from endpoints.analysis import analysis_blueprint
from werkzeug.utils import secure_filename
import os
from QnA_full_class import qnatb
import uuid
import shutil
import pickle

# from QnA_full_class_no_lm import qnatb

qna = qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')

# session not able to save text blob


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

allowed_ext = [".pdf", ".ppt", ".doc"]


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
            print(files)
        except Exception as e:
            print(e)

        session['uid'] = uuid.uuid4().hex

        Folder = os.path.join(app.config['UPLOAD_FOLDER'], session['uid'])
        if not os.path.isdir(Folder):
            os.mkdir(Folder)
        session['Folder'] = Folder

        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(session['Folder'], filename))


                except Exception as e:
                    print(e)
        try:
            names = [os.path.join(session['Folder'], i) for i in os.listdir(session['Folder'])]
            # glob.glob()
            _, _, _, _ = qna.files_processor_tb(names)
            with open(os.path.join(session['Folder'], 'qna'), 'wb') as handle:
                pickle.dump(qna, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)
            print("No readable files")

    return redirect('/')


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


@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data = request.form.get("search")
        with open(os.path.join(session['Folder'], 'qna'), 'rb') as handle:
            qna_loaded = pickle.load(handle)

        responses = qna.get_top_n(search_data, top=10, max_length=7)
        return render_template('search.html', responses=responses)

    except Exception as e:
        print(e)
        print('pickle not loaded')
        return redirect('/')




# analysis blueprint

app.register_blueprint(analysis_blueprint)

# if __name__ == '__main__':
#    app.run()
