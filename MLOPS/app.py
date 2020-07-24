import sys
import os
import glob
import re

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        os.system("python3 yolo.py --image {} --yolo yolo-coco".format("static/"+f.filename))
        result=file_path
        return result
    return None
@app.route('/new')
def new():
	with open("result.txt") as f:
		k=f.readlines()
	return render_template("r.html",d=k)


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
