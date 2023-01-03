from flask import Flask, render_template, request
import os
import pickle

from src.constants import MODEL_PATH
from src.utils import test_pre_processing

app = Flask(__name__, template_folder='src/template')
UPLOAD_FOLDER = 'src/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    csv_files = request.files.getlist("csvfile")
    upload_folder_path = app.config['UPLOAD_FOLDER']
    for file in csv_files:
        file.save(os.path.join(upload_folder_path, file.filename))
    test_data = test_pre_processing(upload_folder_path)
    filename = MODEL_PATH + '/metrodx_model.sav'
    model_pipe = pickle.load(open(filename, 'rb'))
    prediction = model_pipe.predict(test_data)

    replacements = {0: "healthy", 2: "unhealthy"}
    replacer = replacements.get  # For faster gets.

    prediction_txt = [replacer(n, n) for n in prediction]
    table_data = zip(csv_files, prediction_txt)

    return render_template("index.html", table_data=table_data)


if __name__ == '__main__':
    app.run(debug=True)
