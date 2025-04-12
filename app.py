from flask import Flask, render_template, request, jsonify
import os
import wave
import speech_recognition as sr
from pydub import AudioSegment
import pickle
import numpy as np
from textblob import TextBlob
from groq_response import groq_response

app = Flask(__name__)

# Load trained model & vectorizer
with open("model.pkl", "rb") as model_file:
    lr_model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Depression word weights
depression_words = {"suicide": 10, "worthless": 5, "hopeless": 4, "alone": 5, "kill": 9, "die": 19,
                    "sad": 3, "depressed": 5, "lonely": 3, "pain": 6, "suffering": 7, "low": 4}
critical_words = {"suicide", "kill", "worthless", "hopeless", "pain", "suffering"}

def get_weighted_score(text):
    words = text.lower().split()
    score_sum, word_count, critical_detected = 0, 0, False

    for word in words:
        if word in depression_words:
            score_sum += depression_words[word]
            word_count += 1
        if word in critical_words:
            critical_detected = True

    if word_count == 0:
        return 0

    sentiment = TextBlob(text).sentiment.polarity
    if not critical_detected:
        if sentiment > 0.2:
            score_sum *= 0.5
        elif sentiment < -0.2:
            score_sum *= 1.5

    return np.clip((score_sum / max(word_count, 3)) * 2, 0, 10)

def get_depression_level(score):
    if score <= 1:
        return "No Depression"
    elif score <= 2:
        return "Mild Depression"
    elif score <= 4:
        return "Moderate Depression"
    elif score <= 7:
        return "Severe Depression"
    else:
        return "Extreme Depression"

def convert_audio_to_wav(input_file):
    """ Convert any audio format to WAV PCM """
    output_file = "converted_audio.wav"
    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        return None

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    converted_file = convert_audio_to_wav(file_path)
    if not converted_file:
        return "Error: Unable to convert audio"

    with sr.AudioFile(converted_file) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition request failed"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.json['text']
    return analyze_text(user_input)

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({"status": "Error", "response": "No audio file uploaded"})

    audio_file = request.files['audio']
    file_path = "temp_audio_input"

    audio_file.save(file_path)
    transcribed_text = transcribe_audio(file_path)
    os.remove(file_path)

    return analyze_text(transcribed_text)

def analyze_text(user_input):
    vectorized_input = vectorizer.transform([user_input])
    is_depressed = lr_model.predict(vectorized_input)[0]

    if is_depressed == 0:
        response = groq_response(user_input, "No Depression")
        return jsonify({"status": "Not Depressed", "response": response, "depression_score": 0})

    weighted_score = get_weighted_score(user_input)
    depression_level = get_depression_level(weighted_score)
    response = groq_response(user_input, depression_level)

    return jsonify({
        "status": f"Depression Level: {depression_level}",
        "response": response,
        "depression_score": weighted_score
    })

if __name__ == "__main__":
    app.run(debug=True)