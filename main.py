#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
# import numpy as np
import pandas as pd
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


### Text imports ###
from library.text_emotion_recognition import *
from library.text_preprocessor import *
from nltk import *
from tika import parser
from werkzeug.utils import secure_filename
import tempfile



# Flask config
port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'
basedir = os.path.abspath(os.path.dirname(__file__))


# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Read the overall dataframe before the user starts to add his own data
df = pd.read_csv('static/js/db/histo.txt', sep=",")



################################################################################
############################### AUDIO INTERVIEW ################################
################################################################################

# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():

    # Flash message
    flash("After pressing the button above, you will have 15sec to record your audio")
    
    return render_template('audio.html', display_button=False)

# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    data = request.form.get('audio_data', "default_name")
    print("I am running")
    print(data)
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 16 # in sec
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    rec_html_dir = os.path.join('static/audios','voice_recording.wav')
    SER.voice_recording(rec_sub_dir,rec_html_dir, duration=rec_duration)

    # Send Flash message
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audio.html', display_button=True,audio=rec_html_dir)


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
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir

   # rec_sub_dir = os.path.join('static/audios',filenamechanged)
    filepath =  os.path.join('static/audios','voice_recording.wav')
    print(filepath)

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(filepath, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

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
    
    return render_template('audio_dash.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotion_dist, prob_other=emotion_dist_other,emotion_rec=emotion_rec)


################################################################################
############################### TEXT INTERVIEW #################################
################################################################################

global df_text

tempdirectory = tempfile.gettempdir()

@app.route('/text', methods=['POST'])
def text() :
    return render_template('text.html')

# def get_personality(text):
#     try:
#         pred = predict().run(text, model_name = "Personality_traits_NN")
#         return pred
#     except KeyError:
#         return None

def get_text_info(text):
    text = text[0]
    words = wordpunct_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    counts = Counter(words)
    num_words = len(text.split())
    return common_words, num_words, counts

def preprocess_text(text):
    preprocessed_texts = NLTKPreprocessor().transform([text])
    return preprocessed_texts

# @app.route('/text_1', methods=['POST'])
# def text_1():
    
#     text = request.form.get('text')
#     traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
#     probas = get_personality(text)[0].tolist()
    
#     df_text = pd.read_csv('static/js/db/text.txt', sep=",")
#     df_new = df_text.append(pd.DataFrame([probas], columns=traits))
#     df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
#     perso = {}
#     perso['Extraversion'] = probas[0]
#     perso['Neuroticism'] = probas[1]
#     perso['Agreeableness'] = probas[2]
#     perso['Conscientiousness'] = probas[3]
#     perso['Openness'] = probas[4]
    
#     df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
#     df_text_perso = df_text_perso.reset_index()
#     df_text_perso.columns = ['Trait', 'Value']
    
#     df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
#     means = {}
#     means['Extraversion'] = np.mean(df_new['Extraversion'])
#     means['Neuroticism'] = np.mean(df_new['Neuroticism'])
#     means['Agreeableness'] = np.mean(df_new['Agreeableness'])
#     means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
#     means['Openness'] = np.mean(df_new['Openness'])
    
#     probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
#     probas_others = [int(e*100) for e in probas_others]
    
#     df_mean = pd.DataFrame.from_dict(means, orient='index')
#     df_mean = df_mean.reset_index()
#     df_mean.columns = ['Trait', 'Value']
    
#     df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
#     trait_others = df_mean.loc[df_mean['Value'].idxmax()]['Trait']
    
#     probas = [int(e*100) for e in probas]
    
#     data_traits = zip(traits, probas)
    
#     session['probas'] = probas
#     session['text_info'] = {}
#     session['text_info']["common_words"] = []
#     session['text_info']["num_words"] = []
    
#     preprocessed_text = preprocess_text(text)
#     common_words, num_words, counts = get_text_info(preprocessed_text)
    
#     session['text_info']["common_words"].append(common_words)
#     session['text_info']["num_words"].append(num_words)
    
#     trait = traits[probas.index(max(probas))]
    
#     with open("static/js/db/words_perso.txt", "w") as d:
#         d.write("WORDS,FREQ" + '\n')
#         for line in counts :
#             d.write(line + "," + str(counts[line]) + '\n')
#         d.close()
    
#     with open("static/js/db/words_common.txt", "a") as d:
#         for line in counts :
#             d.write(line + "," + str(counts[line]) + '\n')
#         d.close()

#     df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
#     df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
#     df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
#     df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
#     common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

#     df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
#     common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

#     return render_template('text_dash.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

# ALLOWED_EXTENSIONS = set(['pdf'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/text_pdf', methods=['POST'])
# def text_pdf():
#     f = request.files['file']
#     f.save(secure_filename(f.filename))
    
#     text = parser.from_file(f.filename)['content']
#     traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
#     probas = get_personality(text)[0].tolist()
    
#     df_text = pd.read_csv('static/js/db/text.txt', sep=",")
#     df_new = df_text.append(pd.DataFrame([probas], columns=traits))
#     df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
#     perso = {}
#     perso['Extraversion'] = probas[0]
#     perso['Neuroticism'] = probas[1]
#     perso['Agreeableness'] = probas[2]
#     perso['Conscientiousness'] = probas[3]
#     perso['Openness'] = probas[4]
    
#     df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
#     df_text_perso = df_text_perso.reset_index()
#     df_text_perso.columns = ['Trait', 'Value']
    
#     df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
#     means = {}
#     means['Extraversion'] = np.mean(df_new['Extraversion'])
#     means['Neuroticism'] = np.mean(df_new['Neuroticism'])
#     means['Agreeableness'] = np.mean(df_new['Agreeableness'])
#     means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
#     means['Openness'] = np.mean(df_new['Openness'])
    
#     probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
#     probas_others = [int(e*100) for e in probas_others]
    
#     df_mean = pd.DataFrame.from_dict(means, orient='index')
#     df_mean = df_mean.reset_index()
#     df_mean.columns = ['Trait', 'Value']
    
#     df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
#     trait_others = df_mean.ix[df_mean['Value'].idxmax()]['Trait']
    
#     probas = [int(e*100) for e in probas]
    
#     data_traits = zip(traits, probas)
    
#     session['probas'] = probas
#     session['text_info'] = {}
#     session['text_info']["common_words"] = []
#     session['text_info']["num_words"] = []
    
#     preprocessed_text = preprocess_text(text)
#     common_words, num_words, counts = get_text_info(preprocessed_text)
    
#     session['text_info']["common_words"].append(common_words)
#     session['text_info']["num_words"].append(num_words)
    
#     trait = traits[probas.index(max(probas))]
    
#     with open("static/js/db/words_perso.txt", "w") as d:
#         d.write("WORDS,FREQ" + '\n')
#         for line in counts :
#             d.write(line + "," + str(counts[line]) + '\n')
#         d.close()
    
#     with open("static/js/db/words_common.txt", "a") as d:
#         for line in counts :
#             d.write(line + "," + str(counts[line]) + '\n')
#         d.close()

#     df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
#     df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
#     df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
#     df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
#     common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

#     df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
#     common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

#     return render_template('text_dash.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
