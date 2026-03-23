# 🤖 AI Mock Interview Demo

A speech-in / speech-out mock interview platform — powered by Azure OpenAI (LLM) and Azure Speech Services (STT + TTS), orchestrated with LangGraph.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Azure OpenAI resource (deployment created)
- Azure Speech Services resource

### 2. Install Dependencies
```bash
cd AI_Mock_Interview
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configure Credentials
Fill in `.env` with your Azure credentials:
```
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-02-01

AZURE_SPEECH_KEY=<your-key>
AZURE_SPEECH_REGION=<region e.g. eastus>
```

### 4. Run
```bash
python gradio_app.py
```
Open **http://localhost:7860** in your browser.

---

## How It Works

| Step | What happens |
|---|---|
| Upload resume | PDF/DOCX/TXT parsed → top skills extracted via LLM |
| Start interview | LangGraph generates first question → Azure TTS speaks it aloud |
| Record answer | Gradio microphone → Azure STT transcribes → LLM evaluates |
| Feedback | Score (0–100%) + feedback shown after each answer |
| Report card | Final score + per-question breakdown after 5 questions |

---

## Project Structure
```
gradio_app.py          ← Entry point (run this)
app/
  core/config.py       ← Azure credentials (via .env)
  ai/
    langgraph_flow.py  ← Interview workflow (LangGraph)
    llm_client.py      ← Azure OpenAI client
    mcp_server.py      ← MCP tools (parse_resume, extract_skills, evaluate_answer)
  models/
    interview_state.py ← LangGraph state schema
  services/
    resume_parser.py   ← PDF/DOCX/TXT parser
    speech_to_text.py  ← Azure STT
    text_to_speech.py  ← Azure TTS
    scoring.py         ← Score aggregation
```