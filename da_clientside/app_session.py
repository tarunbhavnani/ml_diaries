from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify, Response
#from flask_session import Session #server side session
# from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

import uuid
import pickle
from endpoints.QnA_no_lm import qnatb
from endpoints.functions import PyMuPDF_all, doc_all
import pandas as pd
import shutil

# from QnA_full_class_no_lm import qnatb

app = Flask(__name__)
app.config["SECRET_KEY"] = "OCML3BRawWEUeaxcuKHLpw"
# to use sesison secret key is needed

#app.config["SESSION_TYPE"] = "filesystem" #server side session
#Session(app) #server side session

# qna = qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')
qna = qnatb()

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
        names = [i for i in s if i.split('.')[-1] in ['pdf', 'pptx', 'docx']]
        return render_template('index.html', names=names)
    except:
        return render_template('index.html')


@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    print("Got request in static files")
    # reset files in beginning of session
    reset_files()

    # get files from upload form
    fls = request.files.getlist('files[]')
    print(fls)
    # create a session id, and a respective folder
    session['uid'] = uuid.uuid4().hex
    Folder = os.path.join(app.config['UPLOAD_FOLDER'], session['uid'])
    if not os.path.isdir(Folder):
        os.mkdir(Folder)

    session['Folder'] = Folder
    print(Folder)

    # save all files in the folder, save the responses in json to send
    resp_all = {}

    for f in fls:
        #print(f)
        #print(f.filename)
        if allowed_file(f.filename):
            try:
                filename = secure_filename(f.filename)
                f.save(os.path.join(session['Folder'], filename))
                resp_all[f.filename] = {"success": True, "response": "file saved!"}
            except Exception as e:
                resp_all[filename] = {"success": False, "response": e}
        else:
            resp_all[f.filename] = {"success": False, "response": "Extension type not allowed!"}

    # process files through qna functionality and save pickle
    try:
        names = [os.path.join(session['Folder'], i) for i in os.listdir(session['Folder'])]
        _, _ = qna.files_processor_tb(names)

        session['stats'] = qna.stats()

        with open(os.path.join(session['Folder'], 'qna'), 'wb') as handle:
            pickle.dump(qna, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)
        # app.logger.error(e)
        print("No readable files")
    print(resp_all)
    return jsonify(resp_all), 200


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/delete')
def reset_files():
    # removes files from upload folder and then cleans the session
    try:
        shutil.rmtree(session['Folder'])
        session.pop('Folder', None)
        resp = {"success": True, "response": "Reset"}
    except Exception as e:
        print(e)
        resp = {"success": False, "response": str(e)}

    return jsonify(resp), 200
    #return redirect('/')


@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data = request.form.get("search")
        with open(os.path.join(session['Folder'], 'qna'), 'rb') as handle:
            qna_loaded = pickle.load(handle)

        responses = qna_loaded.get_top_n(search_data, top=10, max_length=7)
        #return render_template('search.html', responses=responses, search_data=search_data)
        resp = {"success": True, "response": responses}
        return jsonify(responses), 200

    except Exception as e:
        resp = {"success": False, "response": str(e)}
        #return redirect('/')
        return jsonify(responses), 200


@app.route('/regex', methods=["GET", "POST"])
def regex():
    try:
        reg_data = request.form.get("search")
        with open(os.path.join(session['Folder'], 'qna'), 'rb') as handle:
            qna_loaded = pickle.load(handle)

        tb_index_reg, overall_dict, docs = qna_loaded.reg_ind(reg_data)

        audit_trail = session['audit_trail']
        audit_trail[reg_data] = overall_dict
        session['audit_trail'] = audit_trail

        tables = []
        for doc in docs:
            try:
                cut = [i for i in tb_index_reg if i['doc'] == doc]
                cut = pd.DataFrame(cut)
                pd.set_option('display.max_colwidth', 40)
                tables.append(cut.to_html(classes='data', justify='left', col_space='100px'))
            except:
                pass

    except Exception as e:
        print(e)
        return redirect('/')
    return render_template("regex.html", overall_dict=overall_dict, tables=tables, reg_data=reg_data, zip=zip)


@app.route('/get_audit_trail')
def get_trail():
    try:
        trail = session['audit_trail']
        fdf = pd.DataFrame()
        for key in trail:
            df = pd.DataFrame(trail[key].items(), columns=['Report', 'Occurance'])
            df['Keyword'] = key
            df = df[['Keyword', 'Report', 'Occurance']]
            fdf = fdf.append(df)
        fdf.to_csv(os.path.join(session['Folder'], "audit_trail.csv"))

        return send_from_directory(session['Folder'], "audit_trail.csv")
    except Exception as e:
        print(e)
        return redirect('/')


@app.route('/uploads/<filename>/')
def upload(filename):
    return send_from_directory(session['Folder'], filename)


@app.route('/metadata/<filename>/')
def metadata(filename):
    print(filename)
    try:
        tables = []
        if filename.endswith('pdf'):
            call_analysis = PyMuPDF_all(session['Folder'], filename)
            md = call_analysis.get_metadata()
        elif filename.endswith('docx'):
            call_analysis = doc_all(session['Folder'], filename)
            md = call_analysis.get_metadata()
        else:
            md = pd.DataFrame(columns=["Parameter", "Details"])

        try:
            print(filename)
            print(session['stats'])
            st = [(i['words'], i['pages']) for i in session['stats'] if i['doc'] == filename]
            md = md.append({"Parameter": "Pages", "Details": st[0][1]}, ignore_index=True)
            md = md.append({"Parameter": "Word-count", "Details": st[0][0]}, ignore_index=True)

        except Exception as e:
            print(e)
            print('something is wrong in metadata')


        md_dict= {i:j for i,j in zip(md.Parameter, md.Details)}
        resp = {"success": False, "response": md_dict}
        #tables.append(md.to_html(classes='data'))

        return jsonify(resp), 200

    except Exception as e:
        resp = {"success": False, "response": str(e)}
        #return redirect('/')
        return jsonify(resp), 200