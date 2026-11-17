import os
import time
import logging

from flask import Flask, render_template, request, jsonify
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# API keys are read only from environment variables.
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Gemini models
# OpenAI models
# No Claude models are included.
ALLOWED_MODELS = [
    {
        "name": "gemini:gemini-3.5-flash",
        "provider": "gemini",
        "model_id": "gemini-3.5-flash",
        "display_name": "Gemini 3.5 Flash"
    },
    {
        "name": "gemini:gemini-3.6-flash",
        "provider": "gemini",
        "model_id": "gemini-3.6-flash",
        "display_name": "Gemini 3.6 Flash"
    },
    {
        "name": "gemini:gemini-3.5-flash-lite",
        "provider": "gemini",
        "model_id": "gemini-3.5-flash-lite",
        "display_name": "Gemini 3.5 Flash Lite"
    },
    {
        "name": "gemini:gemini-3.1-flash-lite",
        "provider": "gemini",
        "model_id": "gemini-3.1-flash-lite",
        "display_name": "Gemini 3.1 Flash Lite"
    },
    {
        "name": "gemini:gemma-4-31b-it",
        "provider": "gemini",
        "model_id": "gemma-4-31b-it",
        "display_name": "Gemma 4 31B"
    },
    {
        "name": "gemini:gemma-4-26b-a4b-it",
        "provider": "gemini",
        "model_id": "gemma-4-26b-a4b-it",
        "display_name": "Gemma 4 26B A4B"
    },
    {
        "name": "openai:gpt-6-astra",
        "provider": "openai",
        "model_id": "gpt-6-astra",
        "display_name": "GPT-6 Astra"
    },
    {
        "name": "openai:gpt-6.1-sol",
        "provider": "openai",
        "model_id": "gpt-6.1-sol",
        "display_name": "GPT-6.1 Sol"
    },
    {
        "name": "openai:gpt-6-luna",
        "provider": "openai",
        "model_id": "gpt-6-luna",
        "display_name": "GPT-6 Luna"
    }
]

MODEL_LOOKUP = {
    item["name"]: item
    for item in ALLOWED_MODELS
}

FAST_SECONDS = 2.5


def maintenance_enabled():
    return os.getenv("MAINTENANCE_MODE", "false").lower() == "true"


def configured_models():
    """Return configured models without sending test requests."""

    configured = []

    if gemini_client is not None:
        configured.extend(
            item for item in ALLOWED_MODELS
            if item["provider"] == "gemini"
        )

    if openai_client is not None:
        configured.extend(
            item for item in ALLOWED_MODELS
            if item["provider"] == "openai"
        )

    return [
        {
            "name": item["name"],
            "provider": item["provider"],
            "display_name": item["display_name"]
        }
        for item in configured
    ]


@app.route("/")
def home():
    if maintenance_enabled():
        return render_template("maintenance.html"), 503

    return render_template("index.html")


@app.route("/models", methods=["GET"])
def models():
    """Return configured models. No model testing is performed."""

    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    return jsonify({
        "models": configured_models()
    })


@app.route("/chat", methods=["POST"])
def chat():

    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance.",
            "unavailable": True
        }), 503

    data = request.get_json(silent=True) or {}

    message = data.get("message", "")
    requested_model = (data.get("model") or "").strip()

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Enter a message.",
            "unavailable": False
        }), 400

    # Validate selected model.
    model_config = MODEL_LOOKUP.get(requested_model)

    if model_config is None:
        return jsonify({
            "error": "This model is not in the ROSE AI model list.",
            "unavailable": True
        }), 400

    provider = model_config["provider"]
    model_id = model_config["model_id"]

    if provider == "gemini" and gemini_client is None:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured.",
            "unavailable": True
        }), 503

    if provider == "openai" and openai_client is None:
        return jsonify({
            "error": "OPENAI_API_KEY is not configured.",
            "unavailable": True
        }), 503

    try:
        started = time.perf_counter()

        # Only the selected model receives this request.
        if provider == "gemini":

            response = gemini_client.models.generate_content(
                model=model_id,
                contents=message
            )

            reply = getattr(response, "text", None)

        elif provider == "openai":

            response = openai_client.responses.create(
                model=model_id,
                input=message
            )

            reply = getattr(response, "output_text", None)

        else:
            return jsonify({
                "error": "Unsupported provider.",
                "unavailable": True
            }), 400

        elapsed = time.perf_counter() - started

        if not reply or not str(reply).strip():
            raise RuntimeError("The selected model returned an empty response.")

        status = "fast" if elapsed <= FAST_SECONDS else "busy"

        return jsonify({
            "reply": str(reply),
            "model": requested_model,
            "status": status
        })

    except Exception:
        app.logger.exception(
            "Chat request failed for provider=%s model=%s",
            provider,
            model_id
        )

        return jsonify({
            "error": (
                "The selected model is unavailable right now. "
                "Check the API key, model access and API billing, "
                "then try again."
            ),
            "unavailable": True
        }), 503


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
