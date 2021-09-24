from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from endpoints.analysis import analysis_blueprint
from werkzeug.utils import secure_filename
import os
from QnA_full_class import qnatb
#from QnA_full_class_no_lm import qnatb

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
    if 'files' in session:
        s = session['files']
        names = [i for i in s if i.endswith(".png") != True]
        return render_template('index.html', names=names)
    else:
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

        documents = []

        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    documents.append(filename)

                except Exception as e:
                    print(e)
        session['files'] = documents

    return redirect('/')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/delete')
def reset_files():
    # removes files from upload folder and then cleans the session
    try:
        s = session['files']
        for file in [i for i in s]:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    except Exception as e:
        print(e)
    session.pop("files", None)
    return redirect('/')


@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data = request.form.get("search")
        s = session['files']
        names = [os.path.join(app.config['UPLOAD_FOLDER'], i) for i in s]
        _, _, _, _ = qna.files_processor_tb(names)
        # responses = qna.get_response_sents(search_data, max_length=7)
        # correct_answer, answer_extracted, _, _ = qna.retrieve_answer(search_data, top=10, max_length=7)
        responses = qna.get_top_n(search_data, top=10, max_length=7)

    except Exception as e:
        print(e)
        return redirect('/')

    return render_template('search.html', responses=responses)


#analysis blueprint

app.register_blueprint(analysis_blueprint)

    

# if __name__ == '__main__':
#    app.run()
