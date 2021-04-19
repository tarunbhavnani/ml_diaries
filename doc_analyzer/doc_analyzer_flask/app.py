import os
import uuid
import fitz
import pandas as pd
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from QnA_full_class import qnatb
#pip install transformers[torch]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
qna= qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')
# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')


# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

allowed_ext = [".pdf", ".ppt", ".doc"]


def allowed_file(file):
    return True in [file.endswith(i) for i in allowed_ext]


@app.route('/')
def index_page():

    names = [i for i in os.listdir(app.config['UPLOAD_FOLDER'])]
    return render_template('index.html', names=names)


@app.route('/delete')
def reset_files():
    names = [i for i in os.listdir(app.config['UPLOAD_FOLDER'])]
    print(names)
    for name in names:
        print(name)
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], name))
    return redirect('/')


@app.route('/', methods=["POST"])
def get_files():
    if request.method == "POST":
        files=request.files.getlist('files[]')
        #user_id = uuid.uuid4().hex
        #UPLOAD_FOLDER_user = os.path.join(UPLOAD_FOLDER, user_id)
        #os.mkdir(UPLOAD_FOLDER_user)
        #print(files)


        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #print(os.listdir(app.config['UPLOAD_FOLDER']))


    return redirect('/')


@app.route("/names")
def print_names():
    names=[i for i in os.listdir(app.config['UPLOAD_FOLDER'])]
    return render_template('names.html', names=names)

@app.route("/analysis")
def all_analysis():

    return render_template('all_analysis.html')



@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


#def process_file( filename):
#    doc = fitz.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#    md = doc.metadata

@app.route('/analysis/<filename>/')
def analysis(filename):
    doc = fitz.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    md = doc.metadata
    df = pd.DataFrame(md.items(), columns=["Paramater", "Details"])
    tables = [df.to_html(classes='data')]
    #titles = df.columns.values
    return render_template("analysis.html", tables=tables, filename=filename)
    

@app.route('/search',methods=["GET", "POST"])
def search():
    search_data=request.form.get("search")
    names=[i for i in os.listdir(app.config['UPLOAD_FOLDER'])]
    doc = fitz.open(os.path.join(app.config['UPLOAD_FOLDER'], names[0]))
    text_blob = ""
    for num, page in enumerate(doc):
        text = page.getText().encode('utf8')
        text = text.decode('utf8')
        text_blob += text
    qna.vectorize_text(text_blob)
    qna.vectorize_question(search_data)
    correct_answer, answer_extracted, _, _ = qna.retrieve_answer( top=20)
    responses = qna.get_response_sents(20)

    return render_template('search.html', text=[correct_answer], responses=responses)

if __name__ == "__main__":
    app.run(port=3000)
