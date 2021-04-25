import os
import uuid
import fitz
import pandas as pd
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from QnA_full_class import qnatb

# pip install transformers[torch]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
qna = qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')
# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')


# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

allowed_ext = [".pdf", ".ppt", ".doc"]


#make a directory for user folders
user_id = "tb"

if not os.path.isdir(os.path.join(UPLOAD_FOLDER,user_id)):
    os.mkdir(os.path.join(UPLOAD_FOLDER,user_id))



def allowed_file(file):
    return True in [file.endswith(i) for i in allowed_ext]


@app.route('/')
def index_page():
    try:
        names = [i for i in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id))]
    except Exception as e:
        print(e)
    return render_template('index.html', names=names)


@app.route('/delete')
def reset_files():

    try:
        names = [i for i in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id))]
        print(names)

        for name in names:
            print(name)
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], user_id, name))
    except Exception as e:
        print(e)
    return redirect('/')


@app.route('/', methods=["POST"])
def get_files():
    if request.method == "POST":
        files = request.files.getlist('files[]')
        print(files)
        global user_id
        user_id = uuid.uuid4().hex
        print(user_id)

        os.mkdir(os.path.join(UPLOAD_FOLDER, user_id))


        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], user_id, filename))
                except Exception as e:
                    print(e)

    return redirect('/')


@app.route("/names")
def print_names():
    names = [i for i in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], user_id))]
    return render_template('names.html', names=names)


@app.route("/analysis")
def all_analysis():
    return render_template('all_analysis.html')


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], user_id), filename)


# def process_file( filename):
#    doc = fitz.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#    md = doc.metadata

@app.route('/analysis/<filename>/')
def analysis(filename):
    try:
        doc = fitz.open(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], user_id), filename))
        md = doc.metadata
        df = pd.DataFrame(md.items(), columns=["Paramater", "Details"])
        tables = [df.to_html(classes='data')]
        # titles = df.columns.values
        return render_template("analysis.html", tables=tables, filename=filename)
    except Exception as e:
        print(e)
        redirect('/')



@app.route('/search', methods=["GET", "POST"])
def search():
    search_data = request.form.get("search")
    names = [i for i in os.listdir(app.config['UPLOAD_FOLDER'])]
    doc = fitz.open(os.path.join(app.config['UPLOAD_FOLDER'], names[0]))
    text_blob = ""
    for num, page in enumerate(doc):
        text = page.getText().encode('utf8')
        text = text.decode('utf8')
        text_blob += text
    qna.vectorize_text(text_blob)
    correct_answer, answer_extracted, _, _ = qna.retrieve_answer(search_data, top=5)

    return render_template('search.html', text=[answer_extracted])


if __name__ == "__main__":
    app.run(port=3000)
