from flask import Blueprint
from flask import current_app
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
import os
import fitz
import pandas as pd

analysis_blueprint= Blueprint("analysis", __name__)

@analysis_blueprint.route('/analysis/<filename>/')
def analysis(filename):
    try:
        doc = fitz.open(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
        md = doc.metadata
        df = pd.DataFrame(md.items(), columns=["Paramater", "Details"])
        tables = [df.to_html(classes='data')]
        # titles = df.columns.values
        return render_template("analysis.html", tables=tables, filename=filename)
    except Exception as e:
        print(e)
        redirect('/')
