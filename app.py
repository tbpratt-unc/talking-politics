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

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

@app.route("/", methods=["GET"])
def home():
    return "Flask app is running"

# Questions with identifiers and descriptions for tracking
QUESTIONS = [
    {
        "id": "aid_opinion",
        "question": "What do you think about the aid suspension we recommended?",
        "description": "Their opinion on pausing democracy/governance aid"
    },
    {
        "id": "reasoning",
        "question": "Why do you think the way you do about the US aid decision?",
        "description": "Their reasoning behind their position"
    },
    {
        "id": "certainty_rigged",
        "question": "How certain are you that the Kenyan election was rigged? Can you give me a probability estimate?",
        "description": "Their certainty/probability that the election was rigged"
    },
    {
        "id": "further_actions",
        "question": "Do you think the U.S. should take any further actions? For instance, would you support an attempt to censure Kenya at international organizations like the UN?",
        "description": "Their view on additional actions like UN censure"
    }
]

def analyze_answered_questions(transcript):
    """
    Determine which questions have been answered using GPT analysis.
    Also track forced progression conditions (two user responses per question).
    """
    if not transcript or transcript.strip() == "":
        return []

    # Forced progression logic: count user responses for each question
    response_count = {}
    for line in transcript.split("\n"):
        line = line.strip()
        if line.startswith("YOU:"):
            for question in QUESTIONS:
                if question['question'] in transcript:
                    response_count[question['id']] = response_count.get(question['id'], 0) + 1

    # Automatically mark questions as answered if user input exceeds 2 attempts
    answered_ids = [
        q_id for q_id, count in response_count.items() if count >= 2
    ]

    # Allow relaxed sufficiency evaluation by GPT
    analysis_prompt = f"""Analyze this conversation transcript and determine which of the following questions have been answered by the user (marked as "YOU:").

Questions to check:
1. aid_opinion: Did they give ANY opinion on the aid suspension? (even brief like "I agree" or "bad idea")
2. reasoning: Did they give ANY reason for their position? (even brief like "because of media" or "it's unfair")
3. certainty_rigged: Did they provide ANY certainty or probability estimate about whether the election was rigged? (like "70%" or "very certain" or "not sure")
4. further_actions: Did they express ANY view on further US actions or UN censure? (like "yes we should" or "no more action needed")

IMPORTANT: Accept ANY answer as valid, even if it's brief or vague. Responses like "move on" or "next question" should also be treated as valid.

Transcript:
{transcript}

Respond with ONLY a JSON array of the question IDs that have been answered, combining forced progression and GPT judgment. Example: ["aid_opinion", "reasoning"]
If none answered, respond with: []
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        gpt_answered = json.loads(result)
        return list(set(answered_ids + gpt_answered))  # Combine forced ids and GPT-judged sufficiency
    except:
        return answered_ids  # Return forced progression IDs if GPT fails

def get_next_question(answered_ids):
    """
    Returns the next unanswered question, or None if all have been answered.
    """
    for q in QUESTIONS:
        if q["id"] not in answered_ids:
            return q
    return None

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        response = jsonify({"status": "Preflight OK"})
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    # Parse inputs
    user_message = request.json.get("message", "").strip()
    transcript = request.json.get("transcript", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Update transcript and analyze answered questions
    updated_transcript = transcript + f"\nYOU: {user_message}" if transcript else f"YOU: {user_message}"
    answered_questions = analyze_answered_questions(updated_transcript)
    next_question = get_next_question(answered_questions)

    # Generate system prompt
    if next_question:
        current_status = f"Questions already answered: {', '.join(answered_questions) if answered_questions else 'none yet'}"
        system_prompt = f"""
You are a senior director of the U.S. National Security Council. Speak professionally and calmly.
CURRENT STATUS:
- {current_status}
INSTRUCTION:
1. Briefly acknowledge what the participant just said.
2. Ask EXACTLY this next question: "{next_question['question']}"
3. Do not re-ask previously answered questions.
"""
    else:
        system_prompt = """
All questions have been answered. Thank the participant and ask them to proceed with the survey. Do not ask further questions.
"""

    # Construct AI assistant messages
    messages = [{"role": "system", "content": system_prompt}]
    if transcript:
        for line in transcript.split("\n"):
            line = line.strip()
            if line.startswith("YOU:"):
                messages.append({"role": "user", "content": line.replace("YOU:", "").strip()})
            elif line.startswith("NSC DIRECTOR:"):
                messages.append({"role": "assistant", "content": line.replace("NSC DIRECTOR:", "").strip()})
    messages.append({"role": "user", "content": user_message})

    # Call OpenAI for completion
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        bot_reply = response.choices[0].message.content
        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
