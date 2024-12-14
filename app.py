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
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Your Flask app is running"

@app.route("/chat", methods=["POST"])
def chat():
    # Retrieve the user's message from the request
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Use the OpenAI API with the updated syntax
        response = client.chat.completions.create(
            model="gpt-4",  # Replace with "gpt-3.5-turbo" if applicable
            messages=[
                {"role": "system", "content": "You are a friend who enjoys debating policy issues."},
                {"role": "user", "content": user_message}
            ]
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