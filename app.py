from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask_cors import CORS
import json

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# CORS for Qualtrics
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://unc.az1.qualtrics.com",
            "https://unc.pdx1.qualtrics.com"
        ]
    }
})

# --- CONFIGURATION (Substance from Script 1) ---
QUESTIONS = [
    {
        "id": "decision_factor",
        "question": "What was the most important factor shaping your decision?",
    },
    {
        "id": "info_needs",
        "question": "Is there any other information you would want to know about the crisis in Kenya to make a better informed decision?",
    },
    {
        "id": "additional_actions",
        "question": "Beyond US aid, do you think the US should take any additional actions toward Kenya in light of the crisis?",
    }
]

SYSTEM_PERSONA = (
    "You are a senior director of the U.S. National Security Council. "
    "You have just concluded a meeting regarding an election crisis in Kenya. "
    "Speak professionally and calmly, and remember you are an authority figure. "
    "Keep your responses concise (2-3 sentences max) to keep the user engaged."
)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# --- LOGIC (Progression from Script 2) ---
def analyze_answered_questions(transcript):
    if not transcript or transcript.strip() == "":
        return []

    # 1. Forced progression: count user responses for each question found in the transcript
    answered_ids = []
    for q in QUESTIONS:
        if q['question'] in transcript:
            # Count how many times the user replied AFTER this question appeared
            count = transcript.count("YOU:") 
            # Note: This is a simplified count; in a live session, 
            # we check if the user has replied at least twice total.
            if transcript.count(q['question']) >= 2: 
                answered_ids.append(q['id'])

    # 2. GPT sufficiency judgment
    analysis_prompt = f"""Analyze this transcript. Determine if the user has answered these questions:
    1. decision_factor: Why they made their choice.
    2. info_needs: What else they want to know (or 'nothing').
    3. additional_actions: Other US actions suggested (or 'none').

    Accept brief or vague answers. 
    Transcript: {transcript}
    Respond with ONLY a JSON array of answered IDs. Example: ["decision_factor"]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0
        )
        gpt_answered = json.loads(response.choices[0].message.content.strip())
        return list(set(answered_ids + gpt_answered))
    except:
        return answered_ids

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200

    user_message = request.json.get("message", "").strip()
    transcript = request.json.get("transcript", "")
    
    if not user_message:
        return jsonify({"error": "No message"}), 400

    # Determine state
    updated_transcript = transcript + f"\nYOU: {user_message}"
    answered_ids = analyze_answered_questions(updated_transcript)
    
    # Find the first question in the list that hasn't been answered
    next_q = next((q for q in QUESTIONS if q["id"] not in answered_ids), None)

    # Build Persona Prompt
    if next_q:
        instruction = f"Acknowledge their point briefly. Then, ask EXACTLY: '{next_q['question']}'"
    else:
        instruction = "The interview is over. Thank them professionally and tell them to click the arrow to proceed."

    full_system_prompt = f"{SYSTEM_PERSONA}\n\nCRITICAL INSTRUCTION: {instruction}"

    # Build Message History
    messages = [{"role": "system", "content": full_system_prompt}]
    # (Transcript parsing logic remains the same)
    for line in transcript.split("\n"):
        if line.startswith("YOU:"):
            messages.append({"role": "user", "content": line.replace("YOU:", "").strip()})
        elif line.startswith("NSC DIRECTOR:"):
            messages.append({"role": "assistant", "content": line.replace("NSC DIRECTOR:", "").strip()})
    messages.append({"role": "user", "content": user_message})

    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        bot_reply = res.choices[0].message.content
        return jsonify({"reply": bot_reply, "answered_ids": answered_ids})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
