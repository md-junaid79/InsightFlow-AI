# api/fastapi_app.py
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline_langchain import run_pipeline

app = FastAPI(title="InsightFlow AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_input(user_text = Form(None),file = File(None)):
    # Handle optional file input; extract filename for content type detection
    file_bytes = None
    filename = None

    if file is not None:
        filename = file.filename  # Needed to route to correct extractor
        file_bytes = await file.read()

    # Delegate to unified pipeline that auto-detects content type
    resp = run_pipeline(
        user_text=user_text or "",
        file_bytes=file_bytes,
        filename=filename,
    )
    return resp
