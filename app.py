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

def get_conversation_stage(transcript, user_message):
    """
    Uses a lightweight LLM call to determine which question to ask next 
    based on what has already been discussed in the transcript.
    Returns an integer index: 0, 1, 2, or 3 (where 3 means done).
    
    Includes a feature to force advancement after a question has been asked 3 times.
    """
    if not transcript and not user_message:
        return 0
    
    # Combine transcript and current message for analysis
    full_context = f"{transcript}\nYOU: {user_message}" if transcript else f"YOU: {user_message}"

    classification_prompt = f"""
    Analyze the following conversation transcript between a USER and an NSC DIRECTOR.
    
    IMPORTANT CONTEXT: The conversation ALWAYS begins with the NSC Director asking Question 1: "{QUESTIONS[0]}". 
    Even if this question is not explicitly visible in the transcript, assume it was asked immediately before the User's first response.

    Determine which of the following mandatory questions the USER has already answered satisfactorily.
    
    The Mandatory Questions are:
    1. {QUESTIONS[0]}
    2. {QUESTIONS[1]}
    3. {QUESTIONS[2]}

    Guidance for Analysis:
    - Q1 Answer Check: The user must provide a *reason, justification, or factor* for their decision. If they provide ANY such context (e.g., "lack of information", "fraud", "corruption", "report"), mark Q1 as ANSWERED.
    - Q2 Answer Check: The user must state *one specific piece of information* they want to know (e.g., "counterterrorism impact", "UN response") OR explicitly state they need *no further information* (e.g., "no", "nothing else"). If either condition is met, mark Q2 as ANSWERED.
    - Q3 Answer Check: The user must suggest *any additional action* the US should take OR explicitly state they *do not recommend* any further action. If either condition is met, mark Q3 as ANSWERED.

    Rules:
    - If the user has NOT answered Question 1, return "0".
    - If the user has answered Q1 but NOT Q2, return "1".
    - If the user has answered Q1 and Q2 but NOT Q3, return "2".
    - If the user has answered all three questions, return "3".
    
    Only return the single digit number (0, 1, 2, or 3).
    
    Transcript:
    {full_context}
    """

    try:
        # Step 1: LLM Classification
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Cheap and fast for classification
            messages=[{"role": "system", "content": "You are a logic engine. Output only a single number."},
                      {"role": "user", "content": classification_prompt}],
            temperature=0
        )
        stage_str = response.choices[0].message.content.strip()
        
        # Ensure we get a valid integer, defaulting to 0
        if stage_str in ["0", "1", "2", "3"]:
            current_stage = int(stage_str)
        else:
            current_stage = 0

        # --- Step 2: Repetition Overrule Logic ---

        # If the LLM thinks we are done (Stage 3), we trust it.
        if current_stage >= 3:
            return 3
        
        # Determine the question text associated with the current stage.
        # If current_stage is 0, we check for repetitions of Q1.
        question_text_to_check = QUESTIONS[current_stage]
        
        # Count how many times this specific question (or the previous one) was asked by the Director.
        # We look for the current stage's question being repeated by the Director.
        
        repetition_count = 0
        
        # Find all director turns in the transcript
        director_lines = [
            line.replace("NSC DIRECTOR:", "").strip() 
            for line in transcript.split("\n") 
            if line.startswith("NSC DIRECTOR:")
        ]
        
        # Check for repetitions of the question text
        for line in director_lines:
            # Simple substring check is usually sufficient for persistent questions
            if question_text_to_check in line:
                repetition_count += 1

        # The initial ask *before* the first user message is assumed (or pre-programmed in Qualtrics).
        # We must add 1 to the count for the initial ask of Q1, but not for subsequent questions
        # as those should be fully logged in the transcript.
        
        # Given your transcript example, the AI repeated Q1. Let's make the check stricter:
        # If the stage is 0, we assume the initial Q1 ask happened (count starts at 1).
        if current_stage == 0 and repetition_count == 0:
            # If transcript is empty, initial ask is counted via pre-programming
            repetition_count = 1
        
        # The threshold is 3 asks (original + 2 repetitions). 
        if repetition_count >= 3:
            # Force advance to the next stage index
            return min(current_stage + 1, 3) # Cap at Stage 3 (Done)

        return current_stage # Return the LLM's classification if repetition threshold isn't met

    except Exception as e:
        # Fallback in case the LLM call fails
        print(f"Error in conversation stage classification: {e}")
        return 0

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

    # --- STEP 1: LOGIC CHECK ---
    # Determine which question we should be asking
    current_stage_index = get_conversation_stage(transcript, user_message)

    # --- STEP 2: CONSTRUCT DYNAMIC SYSTEM PROMPT ---
    # We dynamically build the prompt based on the stage to force the AI to behave
    
    task_instruction = ""
    
    if current_stage_index == 0:
        task_instruction = (
            f"The user has not yet answered the first question. "
            f"Ask them exactly this: '{QUESTIONS[0]}'. "
            "Do not move on until they answer this."
        )
    elif current_stage_index == 1:
        task_instruction = (
            f"The user just answered the first question. Acknowledge their answer briefly, "
            f"then ask the second question: '{QUESTIONS[1]}'."
        )
    elif current_stage_index == 2:
        task_instruction = (
            f"The user just answered the second question. Acknowledge their answer briefly, "
            f"then ask the third question: '{QUESTIONS[2]}'."
        )
    else: # Stage 3
        task_instruction = (
            "The user has answered all questions. Thank them for their time and "
            "tell them to click the arrow below to proceed with the survey. "
            "Do not ask any further questions."
        )

    full_system_prompt = f"{SYSTEM_PERSONA}\n\nCURRENT INSTRUCTION: {task_instruction}"

    # --- INITIAL MESSAGES ---
    messages = [{"role": "system", "content": full_system_prompt}]

    # --- RECONSTRUCT TRANSCRIPT FOR CONTEXT ---
    if transcript:
        for line in transcript.split("\n"):
            line = line.strip()
            if not line: continue
            if line.startswith("YOU:"):
                messages.append({"role": "user", "content": line.replace("YOU:", "").strip()})
            elif line.startswith("NSC DIRECTOR:"):
                messages.append({"role": "assistant", "content": line.replace("NSC DIRECTOR:", "").strip()})

    # Add current message
    messages.append({"role": "user", "content": user_message})

    # --- CALL OPENAI ---
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
