from app import app
from flask import render_template, request, redirect, url_for
import os
import librosa
import pickle
from sklearn import preprocessing
import python_speech_features as mfcc
from tensorflow.python.keras.models import Model, load_model
from tensorflow.compat.v1.keras.backend import set_session
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

print ("loading model")


global model
model = load_model('Model2.h5')


UPLOAD_FOLDER = "/Users/renayrahman/Documents/GitHub/music-genre-recognition-Western-AI/app/static"

app.config["UPLOADS"]= "/Users/renayrahman/Documents/GitHub/music-genre-recognition-Western-AI/app/static/css/file/uploads"
app.config["ALLOWED_EXTENSIONS"]= ["MP3"]


        
    
def allowed_mp3(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/upload", methods=["GET", "POST"])
def upload_mp3():

    if request.method == "POST":

        if request.files:

            mp3 = request.files["mp3"]

            if mp3.filename == " ":
                print("File needs a name")
                return redirect(request.url)

            if not allowed_mp3(mp3.filename):
                print("That extension is not allowed")
                return redirect(request.url)

            else:
                filename = secure_filename(mp3.filename)

                mp3.save(os.path.join(app.config["UPLOADS"],filename))

            print("mp3 saved")

            return redirect(request.url)
            

    return render_template("public/upload_mp3.html")




@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('app/uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('public/index.html')

@app.route('/prediction/<filename>')

def prediction(filename):
    #Step 1
    song_features = np.array([[ 3.49943221e-01,  1.78442045e+03,  2.00265019e+03,  3.80648532e+03,
        8.30663910e-02, -1.13596748e+02,  1.21557297e+02, -1.91588268e+01,
        4.23510284e+01, -6.37645817e+00,  1.86188755e+01, -1.36979122e+01,
        1.53446312e+01, -1.22852669e+01,  1.09804916e+01, -8.32432461e+00,
        8.81066894e+00, -3.66736817e+00,  5.75169086e+00, -5.16276264e+00,
        7.50947773e-01, -1.69193780e+00, -4.09952581e-01, -2.30020881e+00,
        1.21992850e+00]])
    print(song_features.shape)
    #Step 2
    
    prediction = model.predict_classes(song_features)
    print(prediction)
#Step 4
    number_to_class = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    index = np.argmax(prediction)
    predicted_class = number_to_class[index]
#Step 5
    return render_template('public/predict.html', predictions=predicted_class)
