# app.py
import os
from flask import *           # import flask
import resume
import cosine
import pdfminer
from flask_uploads import *
#from werkzeug.utils import secure_filename
#from werkzeug.middleware.shared_data import SharedDataMiddleware
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import Counter

import io
from IPython.display import HTML
import pandas as pd


app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_DOCUMENTS_DEST'] = './'
filename="single-resume.csv"

documents = UploadSet('documents', ALL)
configure_uploads(app, documents)
patch_request_class(app)  # set maximum file size, default is 16MB



class UploadForm(FlaskForm):
    document = FileField(validators=[FileRequired()])
    submit = SubmitField(u'Upload')

@app.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

def file_download_link(filename):
    """Create a Plotly Dash 'A' element sthat downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

     
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = documents.save(form.document.data)
        
        os.path.dirname(os.path.abspath('app.py'))
        file_url = documents.url(filename)
        resume.process(filename)
        df = pd.read_csv('single-resume.csv')
        
        columns = df.columns # for a dynamically created table

        table_d = df.to_json(orient='index')    
        html = HTML(df.to_html())
        output = cosine.process(filename)

        return render_template('index.html',form=form,filename=filename,file_url=file_url,html=html,value=filename,out=output)
        return html(file_download_link(filename))
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url)


'''
@app.route("/")                   # at the end point /
def match():                      # call method hello
    return output                 # which returns top 3 matching job
'''

if __name__ == "__main__":        # on running python app.py
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True,host='0.0.0.0', port=port)                     # run the flask app
