
import os
import time
import logging

from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Keep your API key in Render Environment Variables or your local .env file.
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key) if api_key else None

# Cache the model list for five minutes.
_model_cache = {
    "expires": 0,
    "models": []
}

DEFAULT_MODEL_PREFERENCE = "gemini-3.5-flash-lite"


def maintenance_enabled():
    return os.getenv(
        "MAINTENANCE_MODE", "false"
    ).lower() == "true"


def get_available_gemini_models(force_refresh=False):
    """
    Return models available to this API key
    that support generateContent.
    """

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the server."
        )

    now = time.monotonic()

    if (
        not force_refresh
        and now < _model_cache["expires"]
    ):
        return _model_cache["models"]

    available = []

    for model in client.models.list():

        name = getattr(model, "name", None) or ""

        actions = (
            getattr(model, "supported_actions", None)
            or []
        )

        if not name:
            continue

        if "generateContent" not in actions:
            continue

        # Remove the models/ prefix.
        model_id = name.removeprefix("models/")

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


def is_temporary_model_error(exc):
    """
    Identify temporary errors such as quota limits,
    server overload and unavailable services.
    """

    detail = str(exc).lower()

    status_code = (
        getattr(exc, "code", None)
        or getattr(exc, "status_code", None)
    )

    temporary_terms = (
        "resource_exhausted",
        "rate limit",
        "quota",
        "too many requests",
        "unavailable",
        "overloaded",
        "busy",
        "deadline exceeded",
        "timeout",
        "service unavailable",
        "429",
        "503",
        "504"
    )

    return (
        status_code in (429, 502, 503, 504)
        or any(term in detail for term in temporary_terms)
    )


@app.route("/")
def home():

    if maintenance_enabled():
        return render_template(
            "maintenance.html"
        ), 503

    return render_template("index.html")


@app.route("/models", methods=["GET"])
def models():

    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    try:

        force_refresh = (
            request.args.get("refresh") == "1"
        )

        available_models = get_available_gemini_models(
            force_refresh=force_refresh
        )

        return jsonify({
            "models": available_models
        })

    except Exception:

        app.logger.exception(
            "Could not retrieve Gemini models"
        )

        return jsonify({
            "error": "Could not load Gemini models. Check the server API key and try again.",
            "unavailable": True
        }), 503


@app.route("/chat", methods=["POST"])
def chat():

    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    try:

        if client is None:
            return jsonify({
                "error": "GEMINI_API_KEY is not configured on the server.",
                "unavailable": False
            }), 500

        data = request.get_json(silent=True) or {}

        message = data.get("message", "")

        requested_model = (
            data.get("model") or ""
        ).strip()

        if (
            not isinstance(message, str)
            or not message.strip()
        ):
            return jsonify({
                "error": "Enter a message.",
                "unavailable": False
            }), 400

        # Check that the selected model is available
        # to this API key.
        available_models = get_available_gemini_models()

        available_ids = {
            item["name"]
            for item in available_models
        }

        if not available_ids:
            return jsonify({
                "error": "UNAVAILABLE NOW TRY LATER",
                "unavailable": True
            }), 503

        # Do not silently switch to another model.
        if requested_model not in available_ids:
            return jsonify({
                "error": "The selected model is no longer available. Please choose another model.",
                "unavailable": True
            }), 503

        # Send the message to the exact selected model.
        response = client.models.generate_content(
            model=requested_model,
            contents=message
        )

        reply = (
            getattr(response, "text", None)
            or "The model returned an empty response."
        )

        return jsonify({
            "reply": reply,
            "model": requested_model
        })

    except Exception as exc:

        app.logger.exception(
            "Chat request failed"
        )

        if is_temporary_model_error(exc):

            return jsonify({
                "error": "UNAVAILABLE NOW TRY LATER",
                "unavailable": True
            }), 503

        return jsonify({
            "error": "The selected model could not complete this request. Please try another model or try again later.",
            "unavailable": False
        }), 502


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
