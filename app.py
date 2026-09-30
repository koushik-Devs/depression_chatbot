from flask import Flask, render_template, request, jsonify
import os
import tempfile
from pathlib import Path
import wave
import speech_recognition as sr
from pydub import AudioSegment
import pickle
import numpy as np
from textblob import TextBlob
from groq_response import groq_response

app = Flask(__name__)

# Run `python Training.py` once to create these trusted local artifacts.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    raise RuntimeError("Model artifacts are missing. Run 'python Training.py' first.")
with MODEL_PATH.open("rb") as model_file:
    lr_model = pickle.load(model_file)
with VECTORIZER_PATH.open("rb") as vec_file:
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

def convert_audio_to_wav(input_file, output_file):
    """Convert an audio file to WAV PCM at a caller-provided temporary path."""
    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        return None

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            converted_file = os.path.join(temp_dir, "converted.wav")
            if not convert_audio_to_wav(file_path, converted_file):
                return "Error: Unable to convert audio"
            with sr.AudioFile(converted_file) as source:
                audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition request failed"
    except (OSError, ValueError):
        return "Error: Unable to read audio file"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("text")
    if not isinstance(user_input, str) or not user_input.strip():
        return jsonify({"status": "Error", "response": "Text is required."}), 400
    return analyze_text(user_input.strip())

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({"status": "Error", "response": "No audio file uploaded"})

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"status": "Error", "response": "Choose an audio file."}), 400

    suffix = Path(audio_file.filename).suffix[:10]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
        audio_file.save(temp_path)
        transcribed_text = transcribe_audio(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if transcribed_text.startswith("Error:") or transcribed_text in {
        "Could not understand audio", "Speech recognition request failed"
    }:
        return jsonify({"status": "Error", "response": transcribed_text}), 400
    return analyze_text(transcribed_text)

def analyze_text(user_input):
    vectorized_input = vectorizer.transform([user_input])
    is_depressed = lr_model.predict(vectorized_input)[0]

    if is_depressed == 0:
        try:
            response = groq_response(user_input, "No Depression")
        except RuntimeError:
            return jsonify({"status": "Error", "response": "The response service is not configured."}), 503
        return jsonify({"status": "Not Depressed", "response": response, "depression_score": 0})

    weighted_score = get_weighted_score(user_input)
    depression_level = get_depression_level(weighted_score)
    try:
        response = groq_response(user_input, depression_level)
    except RuntimeError:
        return jsonify({"status": "Error", "response": "The response service is not configured."}), 503

    return jsonify({
        "status": f"Depression Level: {depression_level}",
        "response": response,
        "depression_score": weighted_score
    })

if __name__ == "__main__":
    app.run(debug=False)