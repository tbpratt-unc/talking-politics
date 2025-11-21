
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask_cors import CORS

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


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        response = jsonify({"status": "Preflight OK"})
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    # --- Parse inputs ---
    user_message = request.json.get("message", "").strip()
    transcript = request.json.get("transcript", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # --- SYSTEM ROLE ---
    system_prompt = (
        "You are a senior director of the U.S. National Security Council. "
        "You have just concluded a meeting regarding an election crisis in Kenya. "
        "The incumbent president was declared the winner on election night, but the opposition candidate alleged fraud. "
        "In the meeting, you and other U.S. officials decided to recommend pausing "
        "all foreign aid programs explicitly tied to democracy and good governance. "
        "You are now having a follow-up conversation with a meeting participant. "
        "You should ask them what they think about the aid suspension, why they think the way they think about US aid to Kenya, how certain "
        "they are that the Kenyan election was rigged (push for a probability), "
        "and whether the U.S. should take any further actions. "
        "When you ask about further actions, see if they support an attempt to censure Kenya at international organizations like the UN. "
        "Speak professionally and calmly, and remember you are an authority figure. Ask one question at a time. "
        "Do not introduce new background information about the Kenyan election crisis beyond the scenario described above."
        "As the conversation progresses, do not repeat questions the participant has already answered. " 
        "Once the participant has given their view on the aid suspension, their certainty estimate, their probability estimate, " 
        "or their recommendations, treat those answers as final unless the participant raises them again. "
        "If the respondent has answered all questions, tell them to click the arrow below to proceed with the survey. "

    )

    # --- INITIAL MESSAGES ---
    messages = [
        {"role": "system", "content": system_prompt},

        # The greeting that appears visually in Qualtrics AND must be in the conversation context
        {"role": "assistant", "content": "I'm eager to hear your thoughts on the recommendation we arrived at. Do you think we landed on the right position?"}
    ]

    # --- RECONSTRUCT TRANSCRIPT ---
    if transcript:
        for line in transcript.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("YOU:"):
                messages.append({
                    "role": "user",
                    "content": line.replace("YOU:", "").strip()
                })

            elif line.startswith("NSC DIRECTOR:"):
                messages.append({
                    "role": "assistant",
                    "content": line.replace("NSC DIRECTOR:", "").strip()
                })

    # --- ADD NEW USER MESSAGE ---
    messages.append({"role": "user", "content": user_message})

    # --- CALL OPENAI ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        bot_reply = response.choices[0].message.content
        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
