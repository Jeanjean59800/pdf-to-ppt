import os

import google.generativeai as genai
from src.run_metrics import run_metrics


def generate_text(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    run_metrics.increment("api.gemini.generate_text.calls")
    with run_metrics.timed("api.gemini.generate_text.seconds"):
        response = model.generate_content([prompt])
    return response.text
