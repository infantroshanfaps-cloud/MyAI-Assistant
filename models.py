
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

try:
    print("Checking available models...")

    for model in client.models.list():
        print(model.name)

except Exception as e:
    print("ERROR:", repr(e))

finally:
    client.close()
