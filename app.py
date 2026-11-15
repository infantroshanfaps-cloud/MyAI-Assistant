
from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def maintenance_enabled():
    return os.getenv(
        "MAINTENANCE_MODE", "false"
    ).lower() == "true"


@app.route("/")
def home():

    if maintenance_enabled():
        return render_template("maintenance.html"), 503

    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    if maintenance_enabled():
        return jsonify({
            "error": "Brownie AI is under maintenance."
        }), 503

    try:
        data = request.get_json()
        message = data.get("message", "")

        if not message.strip():
            return jsonify({"error": "Enter a message"}), 400

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=message
        )

        return jsonify({"reply": response.text})

    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "Something went wrong"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
