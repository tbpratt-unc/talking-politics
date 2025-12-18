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
# The specific questions from your image
QUESTIONS = [
    "What was the most important factor shaping your decision?",
    "Is there any other information you would want to know about the crisis in Kenya to make a better informed decision?",
    "Beyond US aid, do you think the US should take any additional actions toward Kenya in light of the crisis?"
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

@app.route("/", methods=["GET"])
def home():
    return "Flask app is running"

def get_conversation_stage(transcript, user_message, current_stage_index):
    if not transcript and not user_message:
        return 0
    
    # Count how many times the USER has spoken
    user_turn_count = transcript.count("YOU:") + 1 

    # --- NEW: Hard-coded advancement rules ---
    # Force Stage 1 if user has sent 2+ messages
    # Force Stage 2 if user has sent 4+ messages
    if user_turn_count >= 2:
        return 1 # Move to Q2
    if user_turn_count >= 4:
        return 2 # Move to Q3
    
    full_context = f"{transcript}\nYOU: {user_message}" if transcript else f"YOU: {user_message}"

    classification_prompt = f"""
    Analyze the transcript. Determine if the USER has answered the questions below.
    
    Current Stage Index: {current_stage_index}
    Mandatory Questions:
    0. {QUESTIONS[0]}
    1. {QUESTIONS[1]}
    2. {QUESTIONS[2]}

    Rules:
    - If the user has answered the question for the current stage, suggest the NEXT stage.
    - NEVER return a number lower than {current_stage_index}.
    - Q1 is answered if they provide any reason or mention any factor.
    - Q2 is answered if they provide some information or say 'none' explicitly.
    - Q3 is answered if they suggest an action or say 'none'.

    Only return the single digit number.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a logic engine. Output only a single number."},
                      {"role": "user", "content": classification_prompt}],
            temperature=0
        )
        new_stage = int(response.choices[0].message.content.strip())
        
        # --- FIX 1: The Ratchet (Only move forward) ---
        stage_to_use = max(new_stage, current_stage_index)

        if stage_to_use >= 3:
            return 3
        
        # --- FIX 2: Enhanced Repetition Check ---
        question_text_to_check = QUESTIONS[stage_to_use]
        repetition_count = transcript.count(question_text_to_check)
        
        if stage_to_use == 0:
            repetition_count += 1 # Account for initial Qualtrics ask
        
        if repetition_count >= 3:
            return min(stage_to_use + 1, 3)  # Force progression after 3 repeats

        return stage_to_use

    except Exception as e:
        print(f"Error: {e}")
        return current_stage_index

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        response = jsonify({"status": "Preflight OK"})
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    # --- 1. Parse inputs ---
    user_message = request.json.get("message", "").strip()
    transcript = request.json.get("transcript", "")
    
    # Receive the current stage from Qualtrics (default to 0 if not sent)
    previous_stage = int(request.json.get("current_stage_index", 0))

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # --- 2. Determine Next Stage (Using the Ratchet) ---
    current_stage_index = get_conversation_stage(transcript, user_message, previous_stage)

    # --- 3. Construct Instructions with Strict Constraints ---
    if current_stage_index == 0:
    # Check if the question was already asked to decide the tone
        if QUESTIONS[0] in transcript:
            task_instruction = (
                "The user provided an answer, but it lacked sufficient detail. "
                "Briefly acknowledge their point about the monitor report, "
                "but ask them to explain WHY that specific report was the most "
                "important factor for their decision."
            )
        else:
            task_instruction = f"Ask the user exactly this: '{QUESTIONS[0]}'."
    
    elif current_stage_index == 1:
        task_instruction = (
            "The user has completed the first topic. "
            "DO NOT ask about their decision factors again. "
            f"Acknowledge their last point and ask exactly: '{QUESTIONS[1]}'."
        )
    elif current_stage_index == 2:
        task_instruction = (
            "The user has completed the first two topics. "
            "DO NOT go back to previous questions. "
            f"Acknowledge and ask exactly: '{QUESTIONS[2]}'."
        )
    else:
        task_instruction = (
            "The interview is over. Do not ask any more questions. "
            "Thank them and tell them to click the arrow to proceed."
        )

    full_system_prompt = (
        f"{SYSTEM_PERSONA}\n"
        "CRITICAL RULE: You must proceed linearly. Once a topic is discussed, "
        "never refer back to it or re-ask previous questions. "
        f"\n\nCURRENT INSTRUCTION: {task_instruction}"
    )

    # --- 4. Reconstruct Messages ---
    messages = [{"role": "system", "content": full_system_prompt}]
    if transcript:
        for line in transcript.split("\n"):
            line = line.strip()
            if not line: continue
            if line.startswith("YOU:"):
                messages.append({"role": "user", "content": line.replace("YOU:", "").strip()})
            elif line.startswith("NSC DIRECTOR:"):
                messages.append({"role": "assistant", "content": line.replace("NSC DIRECTOR:", "").strip()})

    messages.append({"role": "user", "content": user_message})

    # --- 5. Call OpenAI & Return Stage ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content
        
        # IMPORTANT: Return current_stage_index so Qualtrics can track the state
        return jsonify({
            "reply": bot_reply,
            "current_stage_index": current_stage_index 
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
````
