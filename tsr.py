from flask import Flask, render_template,render_template_string
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,SelectField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import cv2, csv, glob, os, easyocr
import numpy as np
from math import ceil
from author import author_partially_bordered, author_bordered, author_unbordered
from self import html_output

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm_table(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    list=SelectField('List',choices=["Partially Bordered","Unbordered","Bordered"])
    submit = SubmitField("Detect TSR")

class UploadFileForm_img(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Detect Tables")
    



@app.route('/detect_table', methods=['GET',"POST"])
def detect_table():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'])
    files = glob.glob(os.path.join(path,"*"))
    for f in files:
        os.remove(f)  
    form = UploadFileForm_img()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        ext = secure_filename(file.filename).split(".")[-1]
        file.save(os.path.join(path,'detect_tables'+"."+ext)) # Then save the file
        for x in range(2):
            file.save(os.path.join(path,f'table{x}'+"."+ext))
        files = [x.split("/")[-1] for x in glob.glob(os.path.join(path,"*")) if not x.endswith("detect_tables."+ext)]
        return render_template("detect_table.html",form=form,show_tables=True,names=files,ext=ext)
    return render_template("detect_table.html",form=form)


@app.route('/', methods=['GET'])
def home():
  return render_template("home.html")



@app.route('/table/<id>/<ext>/', methods=['GET'])
def detect(id,ext):
    file = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],f'table{id}')
    return render_template_string("""<img src="{{url_for('static', filename='files/table'+id+'.'+ext)}}" />""",ext=ext,id=id)

@app.route('/img_to_html', methods=['GET',"POST"])
def image_to_html():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'])
    files = glob.glob(os.path.join(path,"*"))
    for f in files:
        os.remove(f)
    form = UploadFileForm_table()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        ext = secure_filename(file.filename).split(".")[-1]
        file_path = os.path.join(path,'table_img'+"."+ext)
        file.save(file_path) # Then save the file
        algo_type = (form.list.data)
        if  algo_type == "Unbordered":
            try:
                author_unbordered(file_path) #update file saving in function
            except:
                print("Unable to detect")
        elif algo_type == "Bordered":
            try:
                author_bordered(file_path)
            except:
                print("Unable to detect")
        else:
            try:
                author_partially_bordered(file_path)
            except:
                print("Unable to detect")
        
        try:
            table = html_output(file_path)
        except:
            print("Unable to detect using self")
        return render_template('img_to_table.html', form=form,table=table,ext=ext)
    return render_template('img_to_table.html', form=form,table=False)

if __name__ == '__main__':
    app.run(debug=True)