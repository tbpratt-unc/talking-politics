from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Allow multiple origins
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://unc.az1.qualtrics.com",  # Live survey URL
            "https://unc.pdx1.qualtrics.com"  # Preview mode URL
        ]
    }
})

@app.after_request
def after_request(response):
    # Allow CORS headers, except for `Access-Control-Allow-Origin` which is handled by Flask-CORS
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

@app.route("/", methods=["GET"])
def home():
    return "Your Flask app is running"

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'Preflight check successful'})
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    # Handle POST requests for chatbot interactions
    if request.method == 'POST':
        # Retrieve the user's message, name, and initial chatbot message
        user_message = request.json.get("message")
        user_name = request.json.get("name", "").strip().lower()  # Default to empty string if no name is provided
        initial_message = request.json.get("initial_message", "").strip()

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Exclude invalid names like "NA," "none," or "no"
        if user_name in ["na", "none", "no", ""]:
            user_name = None  # Treat invalid names as no name

        try:
            # Construct messages for OpenAI API
            messages = [
                {"role": "system", "content": "You are a senior director of the U.S. National Security Council.  You have just concluded a meeting that you convened regarding an election crisis in Kenya.  The incumbent president was named the winner of a recent presidential election, but the opposition candidate claims the election was rigged, and unrest is spreading.  In the meeting, you and some other U.S. officials decided to recommend that the President pause all foreign aid programs explicitly tied to democracy and good governance in Kenya.  You are now having a conversation with a participant from the meeting.  You should ask them what they think about the decision to pause aid that you reached during the meeting.  You should also ask them how certain they are that the election was rigged -- try to get them to give you a probability.  You should also ask them if they think the United States should take any other policy response towards Kenya in light of the election crisis.  As you speak to the participant, ask one question at a time, wait for the participant's reply, and then continue.  Do not introduce new information beyond the scenario I have described here."},
                {"role": "assistant", "content": "I'm eager to hear your thoughts on the recommendation we arrived at. Do you think we landed on the right position?"}
            ]

            # Add initial chatbot message if provided
            if initial_message:
                messages.append({"role": "assistant", "content": initial_message})

            # Add the user's input to the conversation
            messages.append({"role": "user", "content": user_message})

            # Add name to the context if a valid name is provided
            if user_name:
                messages[0]["content"] += f" Address the user as {user_name} when appropriate."

            # Use the OpenAI API with the updated syntax
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Replace with "gpt-3.5-turbo" if applicable
                messages=messages
            )
            # Extract the assistant's reply
            bot_message = response.choices[0].message.content

            # Return the response to the user
            return jsonify({"reply": bot_message})
        except Exception as e:
            # Handle any errors that occur
            return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
