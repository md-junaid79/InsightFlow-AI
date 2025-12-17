
import io
import re
from typing import Any, Dict

import pdfplumber
import pytesseract
from groq import Groq
from langchain_core.output_parsers import JsonOutputParser

# from openai import OpenAI
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_groq import ChatGroq
from PIL import Image
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api import (
#     NoTranscriptFound,
#     TranscriptsDisabled,
#     YouTubeTranscriptApi,
# )

from .config import (
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    GROQ_STT_MODEL,
    TESSERACT_LANG,
    YOUTUBE_TRANSCRIPT_LANG,
)


def get_llm(model_name=None):
    model_name = model_name or GROQ_LLM_MODEL
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=0,
        max_tokens=2048,
    )



# 
# 1. Input router 
# 

def route_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide input_type based on uploaded file (if any).
    Expects:
      inputs = {
        "user_text": str | None,
        "file_bytes": bytes | None,
        "filename": str | None
      }
    Returns the same dict + "input_type": "text"|"image"|"pdf"|"audio".
    """
    filename = (inputs.get("filename") or "").lower()
    file_bytes = inputs.get("file_bytes")

    if file_bytes and filename:
        if filename.endswith((".png", ".jpg", ".jpeg")):
            inputs["input_type"] = "image"
        elif filename.endswith(".pdf"):
            inputs["input_type"] = "pdf"
        elif filename.endswith((".mp3", ".wav", ".m4a")):
            inputs["input_type"] = "audio"
        else:
            # default unknown file -> treat as text 
            inputs["input_type"] = "text"
    else:
        inputs["input_type"] = "text"

    # Always ensure user_text is a string
    inputs["user_text"] = inputs.get("user_text") or ""
    return inputs


# 
# 2. Content extraction helpers
# 

def _extract_youtube_id_from_text(text: str) -> str | None:
    """
    Try to find a YouTube video id in the text.
    Supports full URLs and short links.
    """
    # Direct ID pattern fallback
    youtube_id_pattern = r"(?:v=|be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
    match = re.search(youtube_id_pattern, text)
    if match:
        k = match.group(1)
        return k
    return None


def _fetch_youtube_transcript(video_id: str) -> str | None:
    """
    Fetch transcript for a given video_id using youtube-transcript-api.
    Returns raw transcript text or None on failure.
    """
    ytt_api = YouTubeTranscriptApi()
    try:
        fetched_transcript = ytt_api.fetch(video_id)
        snippets = [snippet.text for snippet in fetched_transcript]
        return " ".join(snippets).strip()

    except Exception:
        return None


def extract_text_from_plain(user_text: str) -> Dict[str, Any]:
    """
    - If there is a YouTube URL/id, try to fetch transcript.
    - Otherwise just clean and return the raw text.
    """
    cleaned = user_text.strip()
    video_id = _extract_youtube_id_from_text(cleaned)

    if video_id:
        transcript = _fetch_youtube_transcript(video_id)
        if transcript:
            return {
                "extracted_text": transcript,
                "meta": {"source": "youtube_transcript", "video_id": video_id},
            }
        # Fallback: keeping original text but note failure
        return {
            "extracted_text": cleaned,
            "meta": {
                "source": "plain_text",
                "youtube_transcript_error": "Could not fetch transcript; using plain text only.",
                "video_id": video_id,
            },
        }

    return {"extracted_text": cleaned, "meta": {"source": "plain_text"}}


def extract_text_from_image_bytes(file_bytes: bytes) -> Dict[str, Any]:
    """
    Use Tesseract OCR via pytesseract to read text from an image.
    Returns extracted_text and an approximate confidence.
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale

    # Basic pre-processing

    # Get word-level data so we can compute an approximate confidence
    data = pytesseract.image_to_data(
        image,
        lang=TESSERACT_LANG,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []
    for text, conf in zip(data["text"], data["conf"]):
        if text.strip():
            words.append(text.strip())
        try:
            c = float(conf)
            if c >= 0:  # Tesseract uses -1 for "no confidence"
                confidences.append(c)
        except ValueError:
            continue

    extracted_text = " ".join(words).strip()
    avg_conf = (sum(confidences) / len(confidences)) / 100 if confidences else 0.0

    if not extracted_text:
        extracted_text = "[No readable text detected by OCR]"

    meta = {
        "ocr_confidence": round(avg_conf, 3),
        "source": "image_ocr",
    }
    return {"extracted_text": extracted_text, "meta": meta}

def extract_text_from_pdf_bytes(file_bytes: bytes) -> Dict[str, Any]:
    '''extracting text from pdf using pdfplumber library'''
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    text = "\n".join(text_parts).strip()
    if not text:
        text = "[OCR result placeholder from PDF pages]"
    meta = {"ocr_confidence": 0.75}
    return {"extracted_text": text, "meta": meta}

groq_client = Groq(api_key=GROQ_API_KEY)

def extract_text_from_audio_bytes(file_bytes: bytes) -> Dict[str, Any]:
    """
    Use Groq Whisper for transcription, and pydub for approximate duration.
    """
    buf = io.BytesIO(file_bytes)
    if not hasattr(buf, "name"):
        buf.name = "audio.wav"

    # Duration (seconds) via pydub
    try:
        audio_seg = AudioSegment.from_file(buf)
        duration_sec = round(len(audio_seg) / 1000.0, 2)
        buf.seek(0)  # rewind for STT
    except Exception:
        duration_sec = 0.0
        buf.seek(0)

    try:
        result = groq_client.audio.transcriptions.create(
            file=buf,
            model=GROQ_STT_MODEL,
        )
        text = result.text.strip()
    except Exception as e:
        text = f"[Transcription failed: {e}]"

    return {
        "extracted_text": text,
        "meta": {
            "source": "groq_whisper",
            "duration_sec": duration_sec,
        },
    }



def extract_content(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Switch by input_type and attach extracted_text + meta.
    Keeps user_text & other fields.
    """
    input_type = inputs["input_type"]
    file_bytes = inputs.get("file_bytes")
    user_text = inputs.get("user_text", "")

    if input_type == "text":
        result = extract_text_from_plain(user_text)
    elif input_type == "image":
        result = extract_text_from_image_bytes(file_bytes)
    elif input_type == "pdf":
        result = extract_text_from_pdf_bytes(file_bytes)
    elif input_type == "audio":
        result = extract_text_from_audio_bytes(file_bytes)
    else:
        result = {"extracted_text": user_text, "meta": {}}

    inputs["extracted_text"] = result["extracted_text"]
    inputs["extract_meta"] = result["meta"]
    return inputs


# 
# 3. Intent analysis with LangChain
# 

def build_intent_chain():
    """
    Using LLM to decide:
      - task (summarize/sentiment/code_explain/youtube/generic_qa)
      - needs_clarification (bool)
      - clarification_question (str|None)
    """

    intent_schema = """
    You are an intent classifier for a multi-modal assistant.
    Based ONLY on the user's request and the extracted content,
    choose the task and decide if you must ask a follow-up question.

    Valid tasks:
    - "summarize"   -> summarize the content
    - "sentiment"       -> detect sentiment/mood of the content
    - "code_explain   -> explain code and highlight issues
    - "youtube"         -> summarize a YouTube video from its transcript or URL
    - "generic_qa" -> general question answering over the content

    Mandatory rule:
    - If the request is ambiguous or could mean multiple tasks,
    you MUST set "needs_clarification" to true and ask exactly ONE short question.

    Return a JSON object with keys:
    - task: string
    - needs_clarification: boolean
    - clarification_question: string or null
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", intent_schema),
            ("user", "User request:\n{user_text}\n\nExtracted content:\n{extracted_text}"),
        ]
    )

    llm = get_llm()
    parser = JsonOutputParser()

    chain = prompt | llm | parser
    return chain


# 
# 4. Task handler chains
# 

def build_summarize_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a summarizer. Return a structured JSON summary."),
            (
                "user",
                "Summarize the following content.\n"
                "Return JSON with keys:\n"
                "- one_liner\n- bullets (list of 3 items)\n- paragraph (about 5 sentences)\n\n"
                "Content:\n{extracted_text}",
            ),
        ]
    )
    llm = get_llm()
    parser = JsonOutputParser()
    return prompt | llm | parser

def build_sentiment_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a sentiment analyzer."),
            (
                "user",
                "Analyze the sentiment of the content below.\n"
                "Return JSON with keys:\n"
                "- label (one of 'positive','negative','neutral')\n"
                "- confidence (0-1)\n"
                "- justification (short explanation)\n\n"
                "Content:\n{extracted_text}",
            ),
        ]
    )
    llm = get_llm()
    parser = JsonOutputParser()
    return prompt | llm | parser

def build_code_explain_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a senior engineer explaining code to a beginner."),
            (
                "user",
                "Explain the following code in simple terms.\n"
                "Also point out any obvious bugs or bad practices.\n"
                "Return JSON with keys:\n"
                "- high_level\n- step_by_step\n- issues\n- time_complexity (guess if possible)\n\n"
                "Code:\n{extracted_text}",
            ),
        ]
    )
    llm = get_llm()
    return prompt | llm 

def build_generic_qa_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You answer questions based on the given content."),
            (
                "user",
                "User question:\n{user_text}\n\n"
                "Relevant content:\n{extracted_text}\n\n"
                "Answer concisely in plain text.",
            ),
        ]
    )
    llm = get_llm()
    return prompt | llm  # text output is fine

def build_youtube_chain():
    # Here we assume extracted_text already contains transcript or description
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You summarize YouTube videos from transcripts."),
            (
                "user",
                "The following text is a transcript or description of a YouTube video.\n"
                "Give a clear summary in 5-7 bullet points and a short one-liner.\n\n"
                "{extracted_text}",
            ),
        ]
    )
    llm = get_llm()
    return prompt | llm


# 
# 5. Decision logic using RunnableBranch
# 

def build_task_branch():
    summarize_chain = build_summarize_chain()
    sentiment_chain = build_sentiment_chain()
    code_chain = build_code_explain_chain()
    generic_chain = build_generic_qa_chain()
    youtube_chain = build_youtube_chain()

    def task_selector(inputs: Dict[str, Any]) -> str:
        return inputs["intent"]["task"]

    # Each branch expects a dict with extracted_text/user_text etc.
    branch = RunnableBranch(
        # (condition, runnable)
        (
            lambda x: x["intent"]["task"] == "summarize",
            RunnableLambda(lambda x: summarize_chain.invoke({"extracted_text": x["extracted_text"]})),
        ),
        (
            lambda x: x["intent"]["task"] == "sentiment",
            RunnableLambda(lambda x: sentiment_chain.invoke({"extracted_text": x["extracted_text"]})),
        ),
        (
            lambda x: x["intent"]["task"] == "code_explain",
            RunnableLambda(lambda x: code_chain.invoke({"extracted_text": x["extracted_text"]})),
        ),
        (
            lambda x: x["intent"]["task"] == "youtube",
            RunnableLambda(lambda x: youtube_chain.invoke({"extracted_text": x["extracted_text"]})),
        ),
        RunnableLambda(lambda x: generic_chain.invoke({"extracted_text": x["extracted_text"], "user_text": x["user_text"]})),
    )

    return branch


def clarify_or_execute(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final decision step.
    If needs_clarification -> return mode=clarify
    Else execute the selected task and return mode=result
    """
    intent = inputs["intent"]
    if intent.get("needs_clarification"):
        return {
            "mode": "clarify",
            "clarification_question": intent.get("clarification_question"),
            "result": None,
            "extracted_text": inputs["extracted_text"],
            "extract_meta": inputs.get("extract_meta", {}),
            "plan_log": [
                f"Input type: {inputs.get('input_type')}",
                "Intent unclear -> asking follow-up question.",
            ],
        }

    # Execute task branch
    task_branch = build_task_branch()
    task_result = task_branch.invoke(inputs)

    plan_log = [
        f"Input type: {inputs.get('input_type')}",
        f"Detected task: {intent.get('task')}",
        "Executed task successfully.",
    ]

    return {
        "mode": "result",
        "clarification_question": None,
        "result": task_result,
        "extracted_text": inputs["extracted_text"],
        "extract_meta": inputs.get("extract_meta", {}),
        "plan_log": plan_log,
    }


# 
# 6. Build the full pipeline chain
# 

def build_pipeline():
    """
    LCEL-style graph:
    user input -> route_input -> extract_content -> intent_chain (parallel) -> clarify_or_execute
    """

    intent_chain = build_intent_chain()

    # Step 1: route + extract (pure Python)
    preprocessing = RunnableLambda(route_input) | RunnableLambda(extract_content)

    # Step 2: run intent_chain in parallel with raw fields
    with_intent = preprocessing | RunnableParallel(
        # pass through original fields
        user_text=lambda x: x["user_text"],
        input_type=lambda x: x["input_type"],
        file_bytes=lambda x: x.get("file_bytes"),
        filename=lambda x: x.get("filename"),
        extracted_text=lambda x: x["extracted_text"],
        extract_meta=lambda x: x["extract_meta"],
        intent=lambda x: intent_chain.invoke(
            {
                "user_text": x["user_text"],
                "extracted_text": x["extracted_text"],
            }
        ),
    )

    # Step 3: clarify or execute
    full_chain = with_intent | RunnableLambda(clarify_or_execute)
    return full_chain

PIPELINE_CHAIN = build_pipeline()

def run_pipeline(user_text: str, file_bytes: bytes | None, filename: str | None) -> Dict[str, Any]:
    """
    Helper to call from Streamlit/FastAPI:
    returns dict with:
      - mode: "clarify" or "result"
      - clarification_question
      - result (task output or None)
      - extracted_text
      - extract_meta
      - plan_log
    """
    inputs = {"user_text": user_text, "file_bytes": file_bytes, "filename": filename}
    return PIPELINE_CHAIN.invoke(inputs)
