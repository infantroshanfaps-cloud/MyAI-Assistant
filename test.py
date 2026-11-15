
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=60000,
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=2,
            max_delay=10,
            http_status_codes=[429, 500, 502, 503, 504]
        )
    )
)

try:
    print("Testing Gemini 3.5 Flash Lite...")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Hello! Reply in one sentence."
    )

    print("\nSUCCESS!")
    print("Gemini:", response.text)

except Exception as e:
    print("\nERROR:", repr(e))

finally:
    client.close()
