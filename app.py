import os
import time
import logging

from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Only these six models are available in ROSE AI.
ALLOWED_MODELS = [
    {"name": "gemini-3.5-flash", "display_name": "Gemini 3.5 Flash"},
    {"name": "gemini-3.6-flash", "display_name": "Gemini 3.6 Flash"},
    {"name": "gemini-3.5-flash-lite", "display_name": "Gemini 3.5 Flash Lite"},
    {"name": "gemini-3.1-flash-lite", "display_name": "Gemini 3.1 Flash Lite"},
    {"name": "gemma-4-31b-it", "display_name": "Gemma 4 31B"},
    {"name": "gemma-4-26b-a4b-it", "display_name": "Gemma 4 26B A4B"},
]

ALLOWED_MODEL_IDS = {item["name"] for item in ALLOWED_MODELS}
FAST_SECONDS = 2.5


def maintenance_enabled():
    return os.getenv("MAINTENANCE_MODE", "false").lower() == "true"


@app.route("/")
def home():
    if maintenance_enabled():
        return render_template("maintenance.html"), 503

    return render_template("index.html")


@app.route("/models", methods=["GET"])
def models():
    """Return the fixed model list. No model test requests are sent."""
    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    return jsonify({"models": ALLOWED_MODELS})


@app.route("/chat", methods=["POST"])
def chat():
    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    if client is None:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured on the server.",
            "unavailable": True,
        }), 500

    data = request.get_json(silent=True) or {}

    message = data.get("message", "")
    requested_model = (data.get("model") or "").strip()

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Enter a message.",
            "unavailable": False,
        }), 400

    # Server-side allowlist: only the six configured models are accepted.
    if requested_model not in ALLOWED_MODEL_IDS:
        return jsonify({
            "error": "This model is not in the ROSE AI model list.",
            "unavailable": True,
        }), 400

    try:
        started = time.perf_counter()

        # One request goes only to the model selected by the user.
        response = client.models.generate_content(
            model=requested_model,
            contents=message,
        )

        elapsed = time.perf_counter() - started
        reply = getattr(response, "text", None)

        if not reply:
            raise RuntimeError(
                "The selected model returned an empty response."
            )

        status = "fast" if elapsed <= FAST_SECONDS else "busy"

        return jsonify({
            "reply": reply,
            "model": requested_model,
            "latency_seconds": round(elapsed, 1),
            "status": status,
        })

    except Exception:
        app.logger.exception(
            "Chat request failed for model %s",
            requested_model,
        )

        return jsonify({
            "error": (
                "NOT AVAILABLE · The selected model did not respond. "
                "Please try again later or choose another model."
            ),
            "unavailable": True,
        }), 503


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
