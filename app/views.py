from app import app
from flask import render_template, request, redirect, url_for
import os
import librosa
import pickle
from sklearn import preprocessing
import python_speech_features as mfcc
from keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

print ("loading model")

global set_session
sess = tf.compat.v1.Session()
global model
model = load_model('Model2.h5')
global graph
graph = tf.compat.v1.get_default_graph()

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
        return redirect(url_for('/prediction/<filename>', filename=filename))
    return render_template('public/index.html')

@app.route('/prediction/<filename>')
def extract_features():
    signal, sr = librosa.load ("uploads/<filename>")
    mfccs = librosa.feature.mfcc(signal,n_mfccs=13,sr=sr)
    return mfccs

def prediction(filename):
    #Step 1
    song_features = extract_features()
    
    #Step 2
    with graph.as_default():
      set_session(sess)
      prediction = model.predict_classes(song_features)
      print(prediction)
#Step 4
      number_to_class = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
      index = np.argsort(prediction)
      predictions = {
        "class1":number_to_class[index[9]],
        "class2":number_to_class[index[8]],
        "class3":number_to_class[index[7]],
        "prob1":prediction[index[9]],
        "prob2":prediction[index[8]],
        "prob3":prediction[index[7]],
      }
#Step 5
    return render_template('predict.html', predictions=predictions)
