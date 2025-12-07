
# **INSIGHTFLOW-AI🤖 – Multimodal📝 AI Assistant**

A multimodal agent that intelligently extracts, understands and transforms user inputs across **Text, Images, PDFs, Audio and YouTube links**, and performs:

* Summarization (1-liner, bullets, paragraph)
* Sentiment analysis (label + confidence + justification)
* Code explanation (steps, bugs, complexity)
* YouTube transcript summarization / Q&A
* Audio transcription + summary
* Conversational Q&A
* Automatic intent clarification when ambiguous

Built using: **LangChain + Groq (free), Tesseract OCR, pdfplumber, youtube-transcript-api, Streamlit, FastAPI**

---

# **Architecture Diagram**
<p align="center">
  <img src="images\architecture.png" alt="WORKFLOW FLOW"/>
</p>

---

# **✅Features**

### **Supported Inputs**

* Text
* Image OCR (`png`, `jpg`)
* PDF (text extraction )
* Audio (transcription using Groq model= Whisper)
* YouTube URLs (automatic transcript fetch)

### **Automatic Task Selection**

* Intent classification via LangChain
* If ambiguous → **exactly one follow-up clarification**

### **⭐Tasks**

| Task Type     | Output Format                                |
| ------------- | -------------------------------------------- |
| Summarization | One-liner + 3 bullets + 5-sentence paragraph |
| Sentiment     | Label + confidence + justification           |
| Code Explain  | High-level + steps + issues + complexity     |
| YouTube       | Transcript summary or Q&A                    |
| Audio         | Transcript + optional summary                |
| Generic QnA   | Plain text answer                            |

---

# **📌Running Instructions**

## Install Tesseract (System OCR)

**Windows:** [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

**Linux:**

```bash
sudo apt install tesseract-ocr
```

**Mac:**

```bash
brew install tesseract
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Create `.env` in project root:

```
GROQ_API_KEY=your_groq_api_key
```

(Free key: [https://console.groq.com](https://console.groq.com))

---

## **Run Streamlit UI**

```bash
streamlit run frontend/streamlit_app.py
```

App opens automatically:

```
http://localhost:8501
```
The streamlit UI
<p align="center">
  <img src="Screenshot 2025-10-19 161730.png" alt="WORKFLOW FLOW" width="650"/>
</p>
---

## Run FastAPI Server- (optional)

```bash
uvicorn api.fastapi_app:app --reload
```

Docs:

```
http://localhost:8000/docs
```

> Streamlit uses the pipeline locally, but FastAPI `/process` is available to satisfy API requirements.

---


# Notes

* All LLM + STT functionality uses **Groq free API**
* OCR uses **Tesseract (free)**
* Modular LangChain pipeline enables production-style behavior
* Output formatting satisfies “text-only” deliverable
* Both UI and API are functional and submission-ready

---

