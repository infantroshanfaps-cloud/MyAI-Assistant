import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, wait

from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

_model_cache = {"expires": 0, "models": []}
_health_cache = {"expires": 0, "results": {}}
DEFAULT_MODEL_PREFERENCE = "gemini-3.5-flash-lite"
FAST_SECONDS = 2.5
HEALTH_TIMEOUT_SECONDS = 12
HEALTH_CACHE_SECONDS = 60


def maintenance_enabled():
    return os.getenv("MAINTENANCE_MODE", "false").lower() == "true"


def get_available_gemini_models(force_refresh=False):
    """List every API-key-visible model supporting generateContent."""
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured on the server.")

    now = time.monotonic()
    if not force_refresh and now < _model_cache["expires"]:
        return _model_cache["models"]

    available = []
    for model in client.models.list():
        name = getattr(model, "name", None) or ""
        actions = getattr(model, "supported_actions", None) or []
        if not name or "generateContent" not in actions:
            continue

        model_id = name.removeprefix("models/")
        display_name = getattr(model, "display_name", None) or model_id
        available.append({
            "name": model_id,
            "display_name": display_name
        })

    available.sort(key=lambda item: item["name"].lower())
    _model_cache["models"] = available
    _model_cache["expires"] = now + 300
    return available


def probe_model(model_id):
    """Send a tiny real generation request and measure response latency."""
    started = time.perf_counter()

    response = client.models.generate_content(
        model=model_id,
        contents="Reply with exactly OK.",
        config={"max_output_tokens": 2, "temperature": 0},
    )

    elapsed = time.perf_counter() - started

    candidates = getattr(response, "candidates", None) or []
    text = getattr(response, "text", None)

    if not text and not candidates:
        raise RuntimeError("Model returned no response.")

    status = "fast" if elapsed <= FAST_SECONDS else "busy"

    return {
        "status": status,
        "seconds": round(elapsed, 1),
    }


def check_all_models(force_refresh=False):
    """Probe all listed models concurrently; cache results briefly."""
    now = time.monotonic()

    if not force_refresh and now < _health_cache["expires"]:
        return _health_cache["results"]

    models = get_available_gemini_models(
        force_refresh=force_refresh
    )

    if not models:
        return {}

    results = {}
    executor = ThreadPoolExecutor(max_workers=len(models))

    futures = {
        executor.submit(probe_model, item["name"]): item["name"]
        for item in models
    }

    done, pending = wait(
        futures,
        timeout=HEALTH_TIMEOUT_SECONDS,
    )

    for future in done:
        model_id = futures[future]

        try:
            results[model_id] = future.result()

        except Exception as exc:
            app.logger.warning(
                "Model health check failed for %s: %s",
                model_id,
                exc,
            )

            results[model_id] = {
                "status": "unavailable",
                "seconds": None,
            }

    for future in pending:
        model_id = futures[future]

        results[model_id] = {
            "status": "unavailable",
            "seconds": None,
        }

        future.cancel()

    executor.shutdown(
        wait=False,
        cancel_futures=True
    )

    _health_cache["results"] = results
    _health_cache["expires"] = (
        time.monotonic() + HEALTH_CACHE_SECONDS
    )

    return results


def is_temporary_model_error(exc):
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
        "502",
        "503",
        "504",
    )

    return (
        status_code in (429, 502, 503, 504)
        or any(term in detail for term in temporary_terms)
    )


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
        force_refresh = request.args.get("refresh") == "1"

        return jsonify({
            "models": get_available_gemini_models(
                force_refresh=force_refresh
            )
        })

    except Exception:
        app.logger.exception("Could not retrieve Gemini models")

        return jsonify({
            "error": "Could not load Gemini models.",
            "unavailable": True,
        }), 503


@app.route("/model-health", methods=["GET"])
def model_health():
    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    try:
        force_refresh = request.args.get("refresh") == "1"

        results = check_all_models(
            force_refresh=force_refresh
        )

        return jsonify({
            "results": results
        })

    except Exception:
        app.logger.exception("Model health check failed")

        return jsonify({
            "error": "Could not check model availability.",
            "unavailable": True,
        }), 503


@app.route("/chat", methods=["POST"])
def chat():
    if maintenance_enabled():
        return jsonify({
            "error": "ROSE AI is under maintenance."
        }), 503

    requested_model = ""

    try:
        if client is None:
            return jsonify({
                "error": "GEMINI_API_KEY is not configured on the server.",
                "unavailable": False,
            }), 500

        data = request.get_json(silent=True) or {}

        message = data.get("message", "")
        requested_model = (data.get("model") or "").strip()

        if not isinstance(message, str) or not message.strip():
            return jsonify({
                "error": "Enter a message.",
                "unavailable": False,
            }), 400

        available_models = get_available_gemini_models()
        available_ids = {
            item["name"] for item in available_models
        }

        if requested_model not in available_ids:
            return jsonify({
                "error": "The selected model is not available for this API key.",
                "unavailable": True,
            }), 503

        started = time.perf_counter()

        response = client.models.generate_content(
            model=requested_model,
            contents=message,
        )

        elapsed = time.perf_counter() - started

        reply = getattr(response, "text", None)

        if not reply:
            return jsonify({
                "error": "The selected model returned no text. It may require a different output mode.",
                "unavailable": False,
            }), 502

        actual_status = (
            "fast" if elapsed <= FAST_SECONDS else "busy"
        )

        _health_cache["results"][requested_model] = {
            "status": actual_status,
            "seconds": round(elapsed, 1),
        }

        _health_cache["expires"] = (
            time.monotonic() + HEALTH_CACHE_SECONDS
        )

        return jsonify({
            "reply": reply,
            "model": requested_model,
            "latency_seconds": round(elapsed, 1),
            "status": actual_status,
        })

    except Exception as exc:
        app.logger.exception("Chat request failed")

        if is_temporary_model_error(exc):
            if requested_model:
                _health_cache["results"][requested_model] = {
                    "status": "unavailable",
                    "seconds": None,
                }

                _health_cache["expires"] = (
                    time.monotonic() + HEALTH_CACHE_SECONDS
                )

            return jsonify({
                "error": "NOT AVAILABLE · The model did not respond successfully.",
                "unavailable": True,
            }), 503

        return jsonify({
            "error": "The selected model could not complete this request. Please try another model.",
            "unavailable": False,
        }), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
