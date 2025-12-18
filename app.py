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

# --- CONFIGURATION ---
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
    "Speak professionally and calmly. You are an authority figure. "
    "Keep your responses concise (2-3 sentences max) to keep the user engaged."
)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

def is_current_question_answered(transcript, user_message, current_q):
    """
    Checks ONLY if the specific current question has been addressed.
    This prevents the AI from skipping multiple stages at once.
    """
    # Safety Check: If the question text appears 2+ times, force move on
    if transcript.count(current_q['question']) >= 2:
        return True

    analysis_prompt = f"""
    You are a logic engine. Analyze the user's latest message in the context of the question asked.
    
    Current Question Asked: "{current_q['question']}"
    User's Latest Message: "{user_message}"

    Did the user provide an answer to THIS SPECIFIC question? 
    (Accept brief or vague answers like "none", "not sure", or "I already told you", but they must have replied to this topic).

    Respond with ONLY 'YES' or 'NO'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0
        )
        return "YES" in response.choices[0].message.content.upper()
    except Exception as e:
        print(f"Logic Error: {e}")
        return False

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200

    user_message = request.json.get("message", "").strip()
    transcript = request.json.get("transcript", "")
    
    # Receive the current stage from Qualtrics (0, 1, or 2)
    current_index = int(request.json.get("current_stage_index", 0))

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # --- 1. DETERMINE IF WE MOVE FORWARD (MAX +1) ---
    if current_index < len(QUESTIONS):
        answered = is_current_question_answered(transcript, user_message, QUESTIONS[current_index])
        if answered:
            current_index += 1

    # --- 2. SELECT THE NEXT TASK ---
    if current_index < len(QUESTIONS):
        next_q = QUESTIONS[current_index]
        task_instruction = f"Acknowledge the user's point briefly. Then, ask EXACTLY this question: '{next_q['question']}'"
    else:
        task_instruction = "The interview is over. Thank them and tell them to click the arrow to proceed. Do not ask more questions."

    full_system_prompt = f"{SYSTEM_PERSONA}\n\nCURRENT TASK: {task_instruction}"

    # --- 3. CONSTRUCT PERSONA RESPONSE ---
    messages = [{"role": "system", "content": full_system_prompt}]
    
    # Reconstruct history from transcript for the assistant's context
    if transcript:
        for line in transcript.split("\n"):
            line = line.strip()
            if not line: continue
            if line.startswith("YOU:"):
                messages.append({"role": "user", "content": line.replace("YOU:", "").strip()})
            elif line.startswith("NSC DIRECTOR:"):
                messages.append({"role": "assistant", "content": line.replace("NSC DIRECTOR:", "").strip()})

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content
        
        # Return BOTH the reply and the updated index for Qualtrics to store
        return jsonify({
            "reply": bot_reply,
            "current_stage_index": current_index 
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
