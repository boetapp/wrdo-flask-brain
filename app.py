from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
import openai
import os

app = Flask(__name__)

# Init Firestore DB
cred = credentials.Certificate('firebase-service-account.json')
default_app = initialize_app(cred)
db = firestore.client()
crumbs_ref = db.collection('crumbs')

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def home():
    return 'WRDO Brain is running.'

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data.get('user_id')
    message = data.get('message')

    # Save crumb
    if user_id and message:
        crumbs_ref.add({'user_id': user_id, 'message': message})

    # Optional: Basic command handling
    if message.lower().startswith("/todo"):
        return jsonify({"response": "Task noted. (To-Do list not yet live.)"})

    # Call OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are WRDO, a smart emotional assistant."},
            {"role": "user", "content": message}
        ]
    )

    return jsonify({"response": response.choices[0].message['content']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
