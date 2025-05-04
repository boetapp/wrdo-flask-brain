from flask import Flask, request, jsonify
import openai, os, json, tempfile
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import whisper

# Load ENV vars
openai.api_key = os.environ.get("OPENAI_API_KEY")
hume_api_key = os.environ.get("HUME_API_KEY")
firebase_creds = json.loads(os.environ.get("FIREBASE_CREDS_JSON"))

# Firebase init
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load Whisper model
whisper_model = whisper.load_model("base")

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    if 'audio' in request.files:
        audio_file = request.files['audio']

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_path = tmp.name
            audio_file.save(audio_path)

        # Transcribe
        transcript = whisper_model.transcribe(audio_path)["text"]

        # Emotion detection via Hume
        hume_result = requests.post(
            "https://api.hume.ai/v0/stream/models/voice/emotion",
            headers={"X-Hume-Api-Key": hume_api_key},
            files={"file": open(audio_path, "rb")}
        ).json()
        
        emotion = hume_result.get("predictions", [{}])[0].get("emotions", [{}])[0].get("name", "neutral")

        os.remove(audio_path)  # clean temp file

    else:
        # Text only mode
        transcript = request.json.get("text")
        emotion = "neutral"

    # Generate GPT reply with emotional tone
    prompt = f"Reply to this message with a {emotion} tone: {transcript}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    reply = response['choices'][0]['message']['content']

    # Save Crumb to Firestore
    db.collection("crumbs").add({
        "text": transcript,
        "emotion": emotion,
        "reply": reply,
        "source": "audio" if 'audio' in request.files else "text",
        "created_at": firestore.SERVER_TIMESTAMP
    })

    return jsonify({
        "you_said": transcript,
        "emotion": emotion,
        "wrdo_replied": reply
    })

if __name__ == "__main__":
    app.run(debug=True)

