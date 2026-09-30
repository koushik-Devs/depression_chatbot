import os
from groq import Groq

# Configure this value in the environment; never commit API credentials.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

# Define Groq response function
def groq_response(user_input, depression_level):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    prompt = f"You are a mental health chatbot of India. The user seems to be experiencing '{depression_level}'. Provide a helpful, empathetic,emotional and supportive response to this input: '{user_input}'"

    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a mental health chatbot."},
                  {"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=1000,
        top_p=1
    )

    return completion.choices[0].message.content.strip()
