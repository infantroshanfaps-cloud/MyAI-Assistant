
from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "")

        if not message.strip():
            return jsonify({"error": "Please enter a message"}), 400

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=message
        )

        return jsonify({"reply": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
