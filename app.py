from flask import Flask, request, jsonify
import os, json, tempfile, requests
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore
import whisper

# ENV VARS
openai_api_key = os.environ.get("OPENAI_API_KEY")
hume_api_key = os.environ.get("HUME_API_KEY")
firebase_creds = json.loads(os.environ.get("FIREBASE_CREDS_JSON"))

# OpenAI Client
client = OpenAI(api_key=openai_api_key)

# Firebase Setup
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Whisper
whisper_model = whisper.load_model("base")

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if 'audio' in request.files:
            audio_file = request.files['audio']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio_path = tmp.name
                audio_file.save(audio_path)

            transcript = whisper_model.transcribe(audio_path)["text"]

            hume_response = requests.post(
                "https://api.hume.ai/v0/stream/models/voice/emotion",
                headers={"X-Hume-Api-Key": hume_api_key},
                files={"file": open(audio_path, "rb")}
            ).json()

            emotion = (
                hume_response.get("predictions", [{}])[0]
                .get("emotions", [{}])[0]
                .get("name", "neutral")
            )

            os.remove(audio_path)
        else:
            transcript = request.json.get("text", "")
            emotion = "neutral"

        # ChatGPT-4o
        prompt = f"Reply with a {emotion} tone to: {transcript}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content

        # Save Crumb
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
