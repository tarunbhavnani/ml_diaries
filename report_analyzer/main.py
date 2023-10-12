

@app.route("/", methods=["POST","GET"])
def index():
    if request.method=="POST":
        if "file" not in request.files:
            app.logger("No file attached")
            return render_template("index.html")
        file= request.files['file']
        if file.filename=="":
            app.logger("No file selected")
            return render_template("index.html")
        if file and allowed_filename(file.filename):
            output= process_file(file)
            return Response(output, headers= {"Content-Disposition":"attachment;filename=%s" %(file.filename)})
    return render_template("index.html")


def process_file(file):
    doc= fitz.open(get_file_extebsion(file.filename), file.read())
    doc= highlight(doc, confusing_words, empty_phrases, white_list)
    return doc.tobytes()


    

Allowed_Extensions={"pdf"}

def get_file_extension(file):
    return file.rsplit(".",1)[1].lower()

def allowed_file(file):
    return "." in file and get_file_extension(file) in Allowed_Extensions