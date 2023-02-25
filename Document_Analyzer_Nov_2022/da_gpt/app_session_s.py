from flask import Flask, render_template, request
import os


from endpoints.functions import Qnatb,get_file_names,process_uploaded_files,delete_files,get_fp_loaded,send_file

app = Flask(__name__)
qna = Qnatb(model_path=r'C:\Users\ELECTROBOT\Desktop\model_dump\minilm-uncased-squad2')

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])


@app.route('/', methods=["GET", "POST"])
def index_page():
    
    file_names= get_file_names()
    
    if request.method == "POST":
        try:
            
            files = request.files.getlist('files[]')
        
            process_uploaded_files(files)
            
            file_names= get_file_names()
            
                    
        except Exception as e:
            print(f"Error occurred while processing files: {e}")
            print("No readable files")
            file_names= get_file_names()

    return render_template('index.html', file_names)



@app.route('/delete')
def reset_files():
    file_names= get_file_names()
    try:
        delete_files()
        file_names= get_file_names()

    except Exception as e:
        print(e)
        print("No resetting")
        file_names= get_file_names()

    return render_template('index.html', file_names)




@app.route('/search', methods=["GET", "POST"])
def search():
    try:
        search_data = request.form.get("search")
        
        fp = get_fp_loaded()
        response_sents= fp.get_response_cosine(search_data)
        
        responses=qna.get_top_n(question=search_data,response_sents=response_sents, top=10)
        
        return render_template('search.html', responses=responses, search_data=search_data)
    
    except Exception as e:
        
        print(f"Error occurred while searching: {e}")
        file_names= get_file_names()
        
        return render_template('index.html', file_names)



@app.route('/uploads/<filename>/')
def upload(filename):
    return send_file(filename)







