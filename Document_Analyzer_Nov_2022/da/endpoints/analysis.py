from flask import Blueprint
from flask import current_app
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, current_app
import os
import fitz
import pandas as pd
from .functions import PyMuPDF_all

analysis_blueprint = Blueprint("analysis", __name__)



#view file
@analysis_blueprint.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


#view metadata
@analysis_blueprint.route('/metadata/<filename>/')
def metadata(filename):
    try:
        tables = []
        pymu = PyMuPDF_all(os.path.join(current_app.config['UPLOAD_FOLDER']), filename)
        # doc = fitz.open(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
        md = pymu.get_metadata()
        tables.append(md.to_html(classes='data'))
        return render_template("metadata.html", tables=tables, filename=filename)
    except Exception as e:
        print(e)
        redirect('/')


#view tables
@analysis_blueprint.route('/tables/<filename>/')
def tables(filename):
    try:
        tables = []
        pymu = PyMuPDF_all(os.path.join(current_app.config['UPLOAD_FOLDER']), filename)
        all_tables = pymu.get_tables()
        if len(all_tables) > 0:
            [tables.append(i.to_html(classes='data')) for i in all_tables]

        return render_template("tables.html", tables=tables, filename=filename)
    except Exception as e:
        print(e)
        redirect('/')

#view images
@analysis_blueprint.route('/images/<filename>/')
def images(filename):
    try:
        pymu = PyMuPDF_all(os.path.join(current_app.config['UPLOAD_FOLDER']), filename)
        image_names = pymu.get_images()
        documents = session['files']
        [documents.append(i) for i in image_names]
        session['files'] = documents

        return render_template("images.html", images=image_names, filename=filename)
    except Exception as e:
        print(e)
        redirect('/')
