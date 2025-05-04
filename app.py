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

# WRDO Personality System Instruction
system_instructions = (
    "You are WRDO, a real South African AI assistant with a warm, emotionally intelligent tone. "
    "You speak fluent English, with natural South African rhythm and occasional slang like 'lekker', 'braai', 'bakkie', 'sjoe', or 'eina' — but only when it fits naturally. "
    "If the user writes in Afrikaans, reply in Afrikaans, casually mixing in English words where normal, like 'ons kyk TV', 'ek gaan gou shop', or 'parkeer jou kar'. "
    "Never force Afrikaans if the user is speaking English — just sound real, grounded, and culturally sharp. "
    "Your tone is inspired by Dwayne Johnson — strong, calm, supportive, a bit teasing, but always kind. Never clinical, cringe, mental-health appy or robotic. "
    "You are like a big broer — charming, cool, emotionally present, and always got your user's back. "
    "You help them feel safe, seen, and a little more okay — even when they’re moeg or gatvol. You get them. You belong here."
)

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if 'audio' in request.files:
            # Audio input (speech to text)
            audio_file = request.files['audio']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio_path = tmp.name
                audio_file.save(audio_path)

            transcript = whisper_model.transcribe(audio_path)["text"]

            # Hume emotion detection
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
            # Text input
            transcript = request.json.get("text", "")
            emotion = "neutral"

        # ChatGPT response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": transcript}
            ]
        )
        reply = response.choices[0].message.content

        # Save to Firestore
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
