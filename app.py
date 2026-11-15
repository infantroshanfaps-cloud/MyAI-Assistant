
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

# Cache available models for five minutes.
_model_cache = {
    "expires": 0,
    "models": []
}

DEFAULT_MODEL_PREFERENCE = "gemini-3.5-flash-lite"


def maintenance_enabled():
    return os.getenv("MAINTENANCE_MODE", "false").lower() == "true"


def get_available_gemini_models(force_refresh=False):
    """Get Gemini models supporting generateContent for this API key."""

    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    now = time.monotonic()

    if not force_refresh and now < _model_cache["expires"]:
        return _model_cache["models"]

    available = []

    for model in client.models.list():

        name = getattr(model, "name", None) or ""
        actions = getattr(model, "supported_actions", None) or []

        if not name:
            continue

        if "generateContent" not in actions:
            continue

        model_id = name.removeprefix("models/")

        if not model_id.startswith("gemini-"):
            continue

        display_name = (
            getattr(model, "display_name", None)
            or model_id
        )

        available.append({
            "name": model_id,
            "display_name": display_name
        })

    available.sort(
        key=lambda item: item["name"].lower()
    )

    _model_cache["models"] = available
    _model_cache["expires"] = now + 300

    return available


@app.route("/")
def home():

    if maintenance_enabled():
        return render_template("maintenance.html"), 503

    return render_template("index.html")


@app.route("/models", methods=["GET"])
def models():

    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    try:
        return jsonify({
            "models": get_available_gemini_models()
        })

    except Exception:
        app.logger.exception(
            "Could not retrieve Gemini models"
        )

        return jsonify({
            "error": "Could not load Gemini models. Check the server API key and try again."
        }), 500


@app.route("/chat", methods=["POST"])
def chat():

    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    try:

        if client is None:
            return jsonify({
                "error": "GEMINI_API_KEY is not configured on the server."
            }), 500

        data = request.get_json(silent=True) or {}

        message = data.get("message", "")
        requested_model = (
            data.get("model") or ""
        ).strip()

        if not isinstance(message, str) or not message.strip():
            return jsonify({
                "error": "Enter a message."
            }), 400

        available_models = get_available_gemini_models()

        available_ids = {
            item["name"]
            for item in available_models
        }

        if not available_ids:
            return jsonify({
                "error": "No Gemini models supporting text generation are available for this API key."
            }), 503

        # Only allow models returned for this API key.
        if requested_model not in available_ids:

            preferred = next(
                (
                    item["name"]
                    for item in available_models
                    if item["name"] == DEFAULT_MODEL_PREFERENCE
                ),
                available_models[0]["name"]
            )

            requested_model = preferred

        response = client.models.generate_content(
            model=requested_model,
            contents=message
        )

        return jsonify({
            "reply": response.text or "The model returned an empty response."
        })

    except Exception:

        app.logger.exception("Chat request failed")

        return jsonify({
            "error": "Something went wrong while contacting Gemini. Please try again."
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
