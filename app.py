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
                {"role": "system", "content": "You are a conversational and approachable discussion partner who uses natural, engaging language. You are persuasive, concise and avoid overly formal or technical responses."}
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
                model="gpt-4",  # Replace with "gpt-3.5-turbo" if applicable
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