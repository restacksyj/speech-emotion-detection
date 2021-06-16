#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
# import numpy as np
import pandas as pd
from library.AudioEmotionRecognition import *
import time
import re
import os
from collections import Counter
import altair as alt

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response, url_for

### Audio imports ###
from library.speech_emotion_recognition import *




# Flask config
port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'
app.config['TEMPLATES_AUTO_RELOAD'] = True
basedir = os.path.abspath(os.path.dirname(__file__))


# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Read the overall dataframe before the user starts to add his own data
df = pd.read_csv('static/js/db/histo.txt', sep=",")



# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():

    # Flash message
    flash("After pressing the button above, you will have 15sec to record your audio")
    
    return render_template('audio.html', display_button=False)

# Audio Recording
# @app.route('/audio_recording', methods=("POST", "GET"))
# def audio_recording():
#
#     # Instanciate new SpeechEmotionRecognition object
#     data = request.form.get('audio_data', "default_name")
#     print("I am running")
#     print(data)
#     SER = speechEmotionRecognition()
#
#     # Voice Recording
#     rec_duration = 16 # in sec
#     rec_sub_dir = os.path.join('tmp','voice_recording.wav')
#     rec_html_dir = os.path.join('static/audios','voice_recording.wav')
#     SER.voice_recording(rec_sub_dir,rec_html_dir, duration=rec_duration)
#
#     # Send Flash message
#     flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")
#
#     return render_template('audio.html', display_button=True,audio=rec_html_dir)


# Audio Emotion Analysis
@app.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():
    filenamechanged=""
    if request.method =="POST":
        print(request.files)
        data = request.files['record']
        data.filename = "voice_recording"
        filenamechanged = data.filename+".wav"
        data.save(os.path.join('static/audios',filenamechanged))

        # redirect(url_for("audio_dash"))

        # return redirect(url_for('/audio_dash'))
    print("I am running")
    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'MODEL_CNN_LSTM.hdf5')
    model_sub_dir_svm = os.path.join('Models')
    print(model_sub_dir)

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)
    SVMSER = AudioEmotionRecognition(model_sub_dir_svm)

    # Voice Record sub dir

   # rec_sub_dir = os.path.join('static/audios',filenamechanged)
    filepath =  os.path.join('static/audios','voice_recording.wav')
    print(filepath)

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(filepath, chunk_step=step*sample_rate)
    svmemotions,svmtimestamp = SVMSER.predict_emotion_from_file(filepath,chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
    # svm_emotion_dist = [int(100 * svmemotions.count(emotion) / len(svmemotions)) for emotion in SVMSER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

    # Get most common emotion of other candidates
    df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")


    major_emotion_other = df_other.EMOTION.mode()[0]

    # Calculate emotion distribution for other candidates
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(os.path.join('static/js/db','audio_emotions_dist_other.txt'), sep=',')
    
    emotion_rec =emotion_dist.sort(reverse=True)

    # Sleep
    time.sleep(0.5)

    emotions_list=[]
    emotion_obj = {'Angry':0, 'Disgust':0, 'Fear':0, 'Happy':0, 'Neutral':0, 'Sad':0, 'Surprise':0}

    for emotion in SER._emotion.values():
        print(len(emotions))
        emotion_obj[emotion] = emotions.count(emotion)
        # print(f'{emotion} {emotions.count(emotion)}')
    print(emotions)
    for emotion_c in emotion_obj.values():
        emotions_list.append(round((emotion_c*100)/len(emotions),2))
    print(emotion_obj)
    print(emotions_list)
    # print(emotion_dist)
    # print(svmemotions)

    return render_template('audio_dash.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotions_list, prob_other=emotion_dist_other,emotion_rec=emotion_rec)



# global df_text
#
# tempdirectory = tempfile.gettempdir()






if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)
