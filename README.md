# Depression Support Chatbot

An educational project that classifies text with a locally trained model and uses Groq to generate supportive replies. It is not a diagnostic tool and should not replace professional care.

## Setup

1. Use Python 3.10 or later and install FFmpeg for audio conversion.
2. Create an environment and install dependencies:

   ```bash
   python -m venv .venv
   pip install -r requirements.txt
   ```

3. Train the local classifier from the included dataset:

   ```bash
   python Training.py
   ```

   This creates `model.pkl` and `vectorizer.pkl` next to the script. They are generated locally and are not committed.

4. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
5. Start the app:

   ```bash
   flask --app app run
   ```

Audio transcription uses the Google Speech Recognition service and requires an internet connection. Do not upload sensitive personal information to a public deployment.
