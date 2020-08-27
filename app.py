# app.py
import os
import io
from flask import *
from flask_uploads import *
from werkzeug.utils import secure_filename

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from IPython.display import HTML

import pandas as pd

# custom modules
import resume
import resume_matching

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
def main():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        job_title = request.form['job_title']
        location = request.form['location']
        return render_template(
            'search.html',
            job_title = job_title,
            location = location
            )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['resume']
        basedir = os.path.abspath(os.path.dirname(__file__))
        filedir = os.path.join(
            basedir, 'uploads', secure_filename(file.filename))
        file.save(filedir)
        df = resume_matching.resume_match(filedir)
        html = HTML(df.to_html())
        return render_template(
            'upload.html',
            html=html
            )

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory('uploads',filename)

# def upload_file():
#     form = UploadForm()
#     if form.validate_on_submit():
#         filename = documents.save(form.document.data)
        
#         os.path.dirname(os.path.abspath('app.py'))
#         file_url = documents.url(filename)

#         df = resume_matching.resume_match(filename)
        
#         columns = df.columns # for a dynamically created table

#         table_d = df.to_json(orient='index')    
#         html = HTML(df.to_html())
#         # output = cosine.process(filename)
#         output='Finished.'

#         return render_template(
#             'index.html',
#             form=form,
#             filename=filename,
#             file_url=file_url,html=html,
#             value=filename#,
#             # out=output
#             )
#         return html(file_download_link(filename))
#     else:
#         file_url = None
#     return render_template('index.html', form=form, file_url=file_url)



'''
@app.route("/")                   # at the end point /
def match():                      # call method hello
    return output                 # which returns top 3 matching job
'''

if __name__ == "__main__":        # on running python app.py
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True,host='0.0.0.0', port=port)                     # run the flask app
