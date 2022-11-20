from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify, Response
#from flask_cors import CORS
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


#qna = qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')
qna= qnatb()


# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Get current path
path = os.getcwd()
print(path)

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
            print(files)
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
            #_, _, _, _ = qna.files_processor_tb(names)
            _, _ = qna.files_processor_tb(names)

            session['stats']=qna.stats()

            with open(os.path.join(session['Folder'], 'qna'), 'wb') as handle:
                pickle.dump(qna, handle, protocol=pickle.HIGHEST_PROTOCOL)
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




@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data= request.form.get("search")
        with open(os.path.join(session['Folder'], 'qna'), 'rb') as handle:
            qna_loaded= pickle.load(handle)

        responses= qna_loaded.get_top_n(search_data, top=10, max_length=7)
        return render_template('search.html', responses=responses, search_data= search_data)
    except Exception as e:
        print(e)
        return redirect('/')


@app.route('/regex', methods= ["GET", "POST"])
def regex():
    try:
        reg_data= request.form.get("search")
        with open(os.path.join(session['Folder'], 'qna'), 'rb') as handle:
            qna_loaded= pickle.load(handle)

        tb_index_reg, overall_dict, docs= qna_loaded.reg_ind(reg_data)

        audit_trail= session['audit_trail']
        audit_trail[reg_data]=overall_dict
        session['audit_trail']=audit_trail

        tables= []
        for doc in docs:
            try:
                cut= [i for i in tb_index_reg if i['doc']==doc]
                cut= pd.DataFrame(cut)
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
        trail= session['audit_trail']
        fdf= pd.DataFrame()
        for key in trail:
            df= pd.DataFrame(trail[key].items(), columns=['Report', 'Occurance'])
            df['Keyword']=key
            df= df[['Keyword','Report', 'Occurance']]
            fdf=fdf.append(df)
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
        tables=[]
        if filename.endswith('pdf'):
            call_analysis= PyMuPDF_all(session['Folder'], filename)
            md= call_analysis.get_metadata()
        elif filename.endswith('docx'):
            call_analysis= doc_all(session['Folder'], filename)
            md= call_analysis.get_metadata()
        else:
            md=pd.DataFrame(columns=["Parameter","Details"])

        try:
            print(filename)
            print(session['stats'])
            st= [(i['words'], i['pages']) for i in session['stats'] if i['doc']==filename]
            md= md.append({"Parameter": "Pages", "Details": st[0][1]}, ignore_index=True)
            md= md.append({"Parameter": "Word-count", "Details": st[0][0]}, ignore_index=True)

        except Exception as e:
            print(e)
            print('something is wrong in metadata')

        tables.append(md.to_html(classes='data'))

        return render_template("metadata.html", tables=tables, filename=filename)

    except:
        return redirect('/')




