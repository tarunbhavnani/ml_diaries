from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify, Response
# from flask_session import Session #server side session
# from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import pickle
from endpoints.QnA_no_lm import qnatb
from endpoints.functions import PyMuPDF_all, doc_all, metadata1
import pandas as pd
import shutil
import json
import re
app = Flask(__name__)
app.config["SECRET_KEY"] = "OCML3BRawWEUeaxcuKHLpw"
# to use sesison secret key is needed

# app.config["SESSION_TYPE"] = "filesystem" #server side session
# Session(app) #server side session

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
        # user.id=<>
        Folder = session['Folder']
        filenames = [i for i in os.listdir(Folder)]
        filenames == [i for i in filenames if i.split('.')[-1] in ['pdf', 'pptx', 'docx']]

        return render_template('index.html', names=filenames)
    except:
        return render_template('index.html')


@app.route('/v1/metadata', methods=['GET'])
def metadata():
    final_response = metadata_service()
    return jsonify(final_response), 200


@app.route('/v1/metadata_web/<filename>', methods=['GET'])
def metadata_web(filename):
    try:
        tables = []
        response = metadata_service()
        final_response = [i for i in response['info'] if i['filename'] == filename][0]
        md = pd.DataFrame(final_response.items(), columns=[":Parameter", "Details"])
        tables.append(md.to_html(classes='data'))
        return render_template("metadata.html", tables=tables, filename=filename)
    except:
        return redirect(url_for('index_page'))


def metadata_service():
    try:
        # user.id=
        Folder = session['Folder']
        with open(os.path.join(Folder, "qna"), 'rb') as handle:
            qna_loaded = pickle.load(handle)
        md = qna_loaded.metadata_all
        response = {"status": "success", "info": md}
    except Exception as e:
        response = {"status": "failed", "info": "metadata service fail"}
    return response


@app.route('/v1/upload', methods=['POST'])
def upload_file():
    final_response = upload_service()
    return jsonify(final_response), 200


@app.route('/v1/upload_web', methods=['POST'])
def upload_file_web():
    final_response = upload_service()
    return redirect(url_for('index_page'))


def upload_service():
    fls = request.files.getlist('filenames')
    session['uid'] = uuid.uuid4().hex
    Folder = os.path.join(app.config['UPLOAD_FOLDER'], session['uid'])
    if not os.path.isdir(Folder):
        os.mkdir(Folder)

    session['Folder'] = Folder

    resp_fail_upload = []
    # fail_counter=0
    for f in fls:
        if allowed_file(f.filename):
            try:
                filename = secure_filename(f.filename)
                f.save(os.path.join(session['Folder'], filename))
                # response={"filename":filename,
                #             "msg": "",
                #             "status":"success"}
                # resp_all.append(response)
            except Exception as e:

                response = {"filename": filename,
                            "msg": str(e),
                            "status": "failed"}
                resp_fail_upload.append(response)
        else:

            response = {"filename": filename,
                        "msg": "Extensiuon type not allowed",
                        "status": "failed"}
            resp_fail_upload.append(response)
    # if fail_counter==0:
    #     status="success"
    # else:
    #     status="failed"

    #    final_response= {"status":status, "info": resp_all}

    qna_resp = get_qna()
    resp_all = qna_resp['info'] + resp_fail_upload

    if "failed" in set([i['status'] for i in resp_all]):
        final_response = {"status": "failed", "info": resp_all}
    else:
        final_response = {"status": "success", "info": resp_all}

    return final_response


def get_qna():
    try:
        # user=<>
        Folder = session['Folder']
        _, _, response_file_processing = qna.files_processor_tb(Folder)
        with open(os.path.join(Folder, "qna"), "wb") as handle:
            pickle.dump(qna, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {"status": "success", "info": response_file_processing}
    except Exception as e:
        return {"status": "failed", "info": ["str(e)"]}


@app.route('/v1/delete', methods=['POST'])
def reset_files():
    final_response = reset_files_service()
    return jsonify(final_response), 200


@app.route('/v1/delete_web', methods=['POST'])
def reset_files_web():
    try:
        final_response = reset_files_service()
        return redirect(url_for('index_page'))
    except:
        return redirect(url_for('index_page'))


def reset_files_service():
    Folder = session['Folder']
    if not os.path.isdir(Folder):
        resp = {"status": "failde", "info": "Folder not present"}
    else:
        try:
            shutil.rmtree(Folder)

            resp = {"status": "success", "info": "reset"}
        except:
            resp = {"status": "success", "info": "error in delete"}

    return resp


@app.route('/v1/search', methods=['POST'])
def search():
    final_response = search_service()
    return jsonify(final_response), 200


@app.route('/v1/search_web', methods=['POST'])
def search_web():
    try:
        final_response = search_service()
        return render_template('search.html', responses=final_response['info'], search_data="search_data")
    except:
        return redirect(url_for('index_page'))


def search_service():
    try:
        # user
        Folder = session['Folder']
        # search_data= request.args.get("search")
        search_data = request.form.get("search")
        with open(os.path.join(Folder, 'qna'), 'rb') as handle:
            qna_loaded = pickle.load(handle)

        responses = qna_loaded.get_top_n(search_data, top=10, max_length=7)
        resp = {"status": "success", "info": responses}

    except Exception as e:
        resp = {"status": "failed", "info": str(e)}
    return resp


@app.route('/regex', methods=["GET", "POST"])
def regex():
    try:
        Folder = session['Folder']
        reg_data = request.form.get("search")
        with open(os.path.join(Folder, 'qna'), 'rb') as handle:
            qna_loaded = pickle.load(handle)

        tb_index_reg, overall_dict, docs = qna_loaded.reg_ind(reg_data)

        #audit_trail = session['audit_trail']
        #audit_trail[reg_data] = overall_dict
        #session['audit_trail'] = audit_trail

        tables = []
        for doc in docs:
            try:
                cut = [i for i in tb_index_reg if i['doc'] == doc]
                cut = pd.DataFrame(cut)
                pd.set_option('display.max_colwidth', 40)
                tables.append(cut.to_html(classes='table table-striped'))
            except:
                pass

    except Exception as e:
        print(e)
        return redirect('/')
    return render_template("regex.html", tb_index_reg=tb_index_reg, overall_dict=overall_dict, tables=tables, reg_data=reg_data, zip=zip)

@app.route('/regex_docs/<dat>')
def regex_docs(dat):
    print(dat)
    dat=re.split("---",dat,maxsplit=1)
    reg_data=dat[0]
    doc = dat[1]
    Folder = session['Folder']

    with open(os.path.join(Folder, 'qna'), 'rb') as handle:
        qna_loaded = pickle.load(handle)

    tb_index_reg, overall_dict, docs = qna_loaded.reg_ind(reg_data)
    final_data= [i for i in tb_index_reg if i['doc']==doc]



    return render_template("regex_docs.html", reg_data=final_data)

@app.route('/v1/get_file/<filename>')
def get_file_add(filename):
    # user
    Folder = session['Folder']
    return send_from_directory(Folder, filename)