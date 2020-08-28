# app.py
import os
import io
from flask import *
# from flask_uploads import *
from werkzeug.utils import secure_filename

from IPython.display import HTML

import pandas as pd

# custom modules
import resume
import resume_matching
import indeed_job_scraper as indeed

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'I have a dream'
# app.config['UPLOADED_DOCUMENTS_DEST'] = './'
# filename="single-resume.csv"

# documents = UploadSet('documents', ALL)
# configure_uploads(app, documents)
# patch_request_class(app)  # set maximum file size, default is 16MB



# class UploadForm(FlaskForm):
#     document = FileField(validators=[FileRequired()])
#     submit = SubmitField(u'Upload')

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
    return render_template(
        'index.html',
        error=''
        )

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        job_title = request.form['job_title']
        location = request.form['location']

        job_dict = indeed.get_indeed_job(job_title, location)

        basedir = os.path.abspath(os.path.dirname(__file__))
        jobdir = os.path.join(
            basedir, 'job_posting', 'job_posting.csv')

        job_df = pd.DataFrame(job_dict).dropna().drop_duplicates()
        job_df.to_csv(jobdir, index=False)

        return render_template(
            'search.html',
            job_title = job_title,
            location = location
            )
    else:
        return render_template(
            'index.html',
            error='An error occurred in searching for job postings.\nPlease enter the search keywords again.'
            )


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['resume']
        basedir = os.path.abspath(os.path.dirname(__file__))
        filedir = os.path.join(
            basedir, 'uploads', secure_filename(file.filename))
        file.save(filedir)

        jobdir = os.path.join(
            basedir, 'job_posting', 'job_posting.csv')

        df = resume_matching.resume_match(filedir,jobdir)
        df.title = '<a href="'+df.url+'">'+df.title+'</a>'
        df.description = df.description\
                         .apply(lambda x : ' '.join(x.replace('\n','')\
                         .split(' ')[:100])+'...')
        html = df[['title','company','location','description']]\
               .to_html(index=False,escape=False)

        return render_template(
            'upload.html',
            html=html
            )
    else:
        return render_template(
            'index.html',
            error='An error occurred in uploading resume.\nPlease enter the search keywords again.'
            )

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory('uploads',filename)



'''
@app.route("/")                   # at the end point /
def match():                      # call method hello
    return output                 # which returns top 3 matching job
'''

if __name__ == "__main__":        # on running python app.py
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True,host='0.0.0.0', port=port)                     # run the flask app
