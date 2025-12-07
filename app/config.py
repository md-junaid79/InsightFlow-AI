# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Free Groq models
GROQ_LLM_MODEL = "llama-3.1-8b-instant"       
GROQ_STT_MODEL = "distil-whisper-large-v3"    # for audio transcription

# OCR + transcripts
TESSERACT_LANG = "eng"
YOUTUBE_TRANSCRIPT_LANG = "en"
