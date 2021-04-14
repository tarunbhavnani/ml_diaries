import os

import fitz
import pandas as pd
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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
        print(files)
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
    




if __name__ == "__main__":
    app.run(port=3000)
