"""
gradio_app.py — AI Mock Interview Demo
Speech-in / Speech-out interview platform powered by:
  • Azure OpenAI  (question generation + answer evaluation)
  • Azure Speech  (STT + TTS)
  • LangGraph     (interview workflow)
"""

import os
import asyncio
import sys
import time

# Disable Gradio's SSRF protection for local backend connections
os.environ.setdefault("SAFEHTTPX_DISABLE", "1")
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator

# ─── Windows Connection Fatigue Fix ──────────────────────────────────────────
if sys.platform == "win32":
    import logging
    # Silence the noisy 'Exception in callback _ProactorBasePipeTransport' errors.
    # The log record MESSAGE is: "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)"
    # The "10054" only appears in the traceback — not the message — so we must match on the transport name.
    class WinErrorFilter(logging.Filter):
        _SUPPRESS = (
            "10054", "forcibly closed", "ConnectionResetError",
            "_ProactorBasePipeTransport", "_call_connection_lost",
        )

        def filter(self, record):
            msg = record.getMessage()
            if any(p in msg for p in self._SUPPRESS):
                return False
            # Also check exception info (covers tracebacks logged as part of the record)
            if record.exc_info and record.exc_info[1]:
                exc_str = str(record.exc_info[1])
                if any(p in exc_str for p in ("10054", "ConnectionResetError", "forcibly closed")):
                    return False
            return True

    # Apply to the asyncio logger (where these errors originate) and other noisy loggers
    for logger_name in ["asyncio", "uvicorn.error", "gradio", "h11", ""]:  # "" = root logger
        logging.getLogger(logger_name).addFilter(WinErrorFilter())

    def silent_exception_handler(loop, context):
        msg = context.get("message")
        exception = context.get("exception")
        exc_str = str(exception) if exception else ""
        msg_str = str(msg) if msg else ""
        
        # Comprehensive check for ConnectionResetError-related strings
        err_keywords = ["WinError 10054", "ConnectionResetError", "forcibly closed", "10054"]
        if any(err in msg_str or err in exc_str for err in err_keywords):
            return
            
        loop.default_exception_handler(context)
    
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(silent_exception_handler)
    except Exception:
        pass

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ─── Ensure FFmpeg for Windows ─────────────────────────────────────────────
if sys.platform == "win32":
    try:
        from static_ffmpeg import add_paths
        add_paths()
        venv_path = os.path.join(os.getcwd(), "venv", "Scripts")
        if os.path.exists(venv_path) and venv_path not in os.environ["PATH"]:
            os.environ["PATH"] = venv_path + os.pathsep + os.environ["PATH"]
        print("✅ FFmpeg paths successfully added")
    except ImportError:
        print("⚠️ static-ffmpeg not found, please run: venv\\Scripts\\pip install static-ffmpeg")

# ─── Local Services ──────────────────────────────────────────────────────────
from app.services.streaming_stt import streaming_stt_service
from app.services.resume_parser import resume_parser_service
from app.services.text_to_speech import tts_service
from app.ai.langgraph_flow import app_graph
from app.ai.mcp_server import extract_skills
import base64

async def _get_audio_b64(text: str) -> str:
    """Helper to generate TTS audio and return as base64 for JS playback."""
    if not text:
        return ""
    try:
        audio_path = tts_service.speak_text(text)
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error generating/encoding audio: {e}")
    return ""

# ─── Custom CSS & JS ──────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Premium High-Interaction Layout ── */
.glass-card {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px);
}
.header-block {
    text-align: center;
    padding: 1.5rem 0;
    background: linear-gradient(90deg, #7c3aed 0%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.video-grid {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.0rem;
}
.video-container {
    flex: 1;
    background: #000 !important;
    border-radius: 20px !important;
    overflow: hidden;
    border: 2px solid rgba(124,58,237,0.3);
    aspect-ratio: 16/9;
}
.ai-avatar-container {
    flex: 1;
    background: radial-gradient(circle at center, #1e1b4b 0%, #0f172a 100%) !important;
    border-radius: 24px !important;
    border: 2px solid rgba(124,58,237,0.4);
    aspect-ratio: 16/9;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.2);
}
.ai-avatar-img {
    width: 45%;
    height: auto;
    z-index: 5;
    filter: drop-shadow(0 0 15px rgba(124, 58, 237, 0.5));
    transition: transform 0.3s ease;
}

/* ── Alexa-like Animated Ring ── */
.speaking-ring {
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    border: 8px solid transparent;
    border-top: 8px solid #60a5fa;
    border-bottom: 8px solid #7c3aed;
    filter: blur(8px);
    opacity: 0.8;
    animation: alexa-spin 2s linear infinite, alexa-pulse 1.5s ease-in-out infinite alternate;
    z-index: 2;
}
@keyframes alexa-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
@keyframes alexa-pulse {
    from { transform: scale(0.95); opacity: 0.4; }
    to { transform: scale(1.1); opacity: 0.9; }
}

/* ── Status Pills ── */
.status-pill {
    padding: 0.4rem 1rem;
    border-radius: 99px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}
.status-listening { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
.status-thinking { background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); animation: status-pulse 1.5s infinite; }
.status-speaking { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
@keyframes status-pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}
"""

# JavaScript to play base64-encoded audio directly in the browser
PLAY_AUDIO_JS = """
    async (b64) => {
        if (!b64) return;
        if (b64 === "STOP") {
            window.aiAudioQueue = [];
            window.aiIsPlaying = false;
            if (window.aiCurrentPlayer) {
                window.aiCurrentPlayer.pause();
                window.aiCurrentPlayer.src = "";
            }
            return;
        }
        if (!window.aiAudioQueue) window.aiAudioQueue = [];
        window.aiAudioQueue.push(b64);
        
        if (window.aiIsPlaying) return;
        
        const playNext = async () => {
            if (window.aiAudioQueue.length === 0) {
                window.aiIsPlaying = false;
                return;
            }
            window.aiIsPlaying = true;
            const nextB64 = window.aiAudioQueue.shift();
            const audio = new Audio("data:audio/wav;base64," + nextB64);
            window.aiCurrentPlayer = audio;
            audio.onended = playNext;
            try {
                await audio.play();
            } catch(e) {
                console.error("Playback failed:", e);
                playNext();
            }
        };
        playNext();
    }
"""

RESET_AUDIO_JS = """
    () => {
        window.aiAudioQueue = [];
        window.aiIsPlaying = false;
        if (window.aiCurrentPlayer) {
            window.aiCurrentPlayer.pause();
            window.aiCurrentPlayer.src = "";
        }
    }
"""

def make_fresh_state():
    return {
        "user_id": "demo_user",
        "resume_text": "",
        "skills": [],
        "questions": [],
        "current_question": None,
        "answers": [],
        "total_score": 0.0,
        "is_finished": False,
        "last_user_input": "",
        "conversation_history": [],
        "is_greeting_done": False,
        "is_awaiting_confirmation": False,
        "buffered_answer": "",
        "difficulty": "Mid-Level",
        "target_questions": 5,
        "last_transcript_time": 0.0,
        "pending_transcript": "",
        "last_question_time": 0.0,
        "last_processed_transcript": "",
        "last_played_audio": "",
        "speaking_until": 0.0,
    }

async def _b64_to_audio(b64_str: Optional[str]) -> Optional[str]:
    """Return the raw b64 string for direct JS playback."""
    return b64_str or ""

async def process_resume(resume_file):
    """Parse resume locally and extract skills."""
    if resume_file is None:
        return "⚠️ No file uploaded.", "", []

    file_path = resume_file.name if hasattr(resume_file, "name") else str(resume_file)
    try:
        # 1. Parse content
        content = resume_parser_service.parse(file_path)
        
        # 2. Extract skills
        skills_json = await extract_skills(content)
        skills = json.loads(skills_json) if skills_json else ["General"]
        
        preview = content[:500] + "..."
        skills_str = " · ".join(f"🔹 {s}" for s in skills)
        return preview, skills_str, skills
    except Exception as e:
        print(f"Error parsing resume: {e}")
        return f"❌ Error: {str(e)}", "", []


async def start_interview(resume_file, skills: list, difficulty: str, num_q: int, session_state: dict):
    """Initialize LangGraph state and trigger first question."""
    resolved_skills = skills if skills else ["General"]
    # resume_file is a file-like object from Gradio or a path string
    file_path = resume_file.name if hasattr(resume_file, "name") else str(resume_file) if resume_file else "resume.pdf"

    new_state = make_fresh_state()
    new_state["skills"] = resolved_skills
    new_state["difficulty"] = difficulty
    new_state["target_questions"] = num_q
    new_state["is_greeting_done"] = True
    new_state["resume_text"] = file_path # In LangGraph this is used as file_path by the first node

    # Start local STT
    streaming_stt_service.add_custom_phrases(resolved_skills)
    streaming_stt_service.start()

    try:
        # First question generation - Run graph until first question
        result = {}
        async for chunk in app_graph.astream({
            "resume_text": file_path,
            "skills": resolved_skills,
            "difficulty": difficulty,
            "questions": [],
            "conversation_history": [],
            "answers": []
        }, config={"recursion_limit": 50}):
            # Each chunk is a dict {node_name: {node_outputs}}
            for node_name, output in chunk.items():
                print(f"--- GRAPH STEP: {node_name} ---")
                status_map = {
                    "resume_parser": "Sarah is reading your resume...",
                    "skill_extractor": "Sarah is identifying your key skills...",
                    "process_turn": "Sarah is preparing the first question..."
                }
                status_txt = status_map.get(node_name, f"Sarah is {node_name}...")
                
                yield (
                    new_state,
                    "Sarah is thinking...",
                    "",
                    f"Round 1 / {num_q}",
                    f"<span class='status-pill status-thinking'>{status_txt}</span>",
                    gr.update(selected=1),
                    True,
                    gr.update(active=True),
                )
                
                result.update(output)
                if node_name == "process_turn":
                    break
            if "process_turn" in chunk:
                break
        
        q_obj = result.get("current_question")
        if q_obj:
            if isinstance(q_obj, dict):
                greeting_text = str(q_obj.get("text", ""))
            else:
                greeting_text = str(getattr(q_obj, "text", ""))
        else:
            greeting_text = "Hello! I'm Sarah. I've reviewed your background. Are you ready to dive into the technical questions?"
            
        audio_b64 = await _get_audio_b64(greeting_text)
        
        # Update session state with graph output
        new_state.update({
            "current_question": q_obj,
            "questions": result.get("questions", [])
        })
        
    except Exception as e:
        print(f"Error starting interview: {e}")
        import traceback
        traceback.print_exc()
        greeting_text = "Hello! I'm Sarah, your AI interviewer. I'm ready to begin. To start, could you please introduce yourself and tell me about your experience?"
        audio_b64 = await _get_audio_b64(greeting_text)

    yield (
        new_state,
        greeting_text,
        audio_b64,
        f"Round 1 / {num_q}",
        "<span class='status-pill status-listening'>🎙 Listening…</span>",
        gr.update(selected=1),
        True,
        gr.update(active=True),
    )


async def submit_answer(audio_input, session_state: dict, override_transcript: str = None):
    """Process user input through LangGraph locally and yield updates."""
    transcript = override_transcript or ""

    if not transcript:
        yield session_state, "⚠️ No transcript.", "Please speak something.", None, None, "–", "<span class='status-pill status-idle'>Waiting…</span>", False, gr.update(), False
        return

    # SIGNAL STOP to the JS Queue only if intentional barge-in or if Sarah is not estimated to be speaking
    curr_time = time.time()
    sarah_speaking = curr_time < session_state.get("speaking_until", 0.0)
    is_long_input = len(transcript.split()) > 3
    
    stop_signal = ""
    if not sarah_speaking or is_long_input:
        stop_signal = "STOP"
        
    yield session_state, transcript, "🤔 Sarah is thinking...", "", stop_signal, gr.update(), "<span class='status-pill status-thinking'>🤔 Sarah is thinking...</span>", False, gr.update(), False

    try:
        # Run LangGraph turn - watch for result in streaming
        result = {}
        async for chunk in app_graph.astream({
            "skills": session_state.get("skills", []),
            "difficulty": session_state.get("difficulty", "Mid-Level"),
            "current_question": session_state.get("current_question"),
            "questions": session_state.get("questions", []),
            "answers": session_state.get("answers", []),
            "last_user_input": transcript,
            "conversation_history": session_state.get("conversation_history", [])
        }, config={"recursion_limit": 50}):
             for node_name, output in chunk.items():
                print(f"--- GRAPH STEP: {node_name} ---")
                status_map = {
                    "process_turn": "Sarah is reviewing and drafting...",
                    "tools": "Sarah is looking up details..."
                }
                status_txt = status_map.get(node_name, "Sarah is thinking...")
                yield session_state, transcript, status_txt, "", "", gr.update(), f"<span class='status-pill status-thinking'>{status_txt}</span>", False, gr.update(), False
                
                result.update(output)
                if node_name == "process_turn":
                    break
             if "process_turn" in chunk:
                break

        # Update local state from result
        session_state.update({
            "answers": result.get("answers", session_state.get("answers", [])),
            "total_score": result.get("total_score", session_state.get("total_score", 0.0)),
            "questions": result.get("questions", session_state.get("questions", [])),
            "current_question": result.get("current_question", session_state.get("current_question"))
        })
        
        # Check if finished
        num_ans = len(session_state["answers"])
        target = session_state.get("target_questions", 5)
        is_done = num_ans >= target
        session_state["is_finished"] = is_done

        # AI Response (Next question or closing)
        q_obj = session_state.get("current_question")
        if q_obj:
            if isinstance(q_obj, dict):
                msg_text = str(q_obj.get("text", ""))
            else:
                msg_text = str(getattr(q_obj, "text", ""))
        else:
            msg_text = "Thank you! That concludes our interview. I'm generating your report now."
        
        if is_done and not msg_text:
             msg_text = "Thank you! That concludes our interview. I'm generating your report now."

        audio_b64 = await _get_audio_b64(msg_text)
        
        # Estimate duration for UI ring
        text_len = len(msg_text)
        estimated_dur = (text_len / 15) + 0.5
        curr_until = session_state.get("speaking_until", 0.0)
        session_state["speaking_until"] = max(curr_until, time.time()) + estimated_dur

        status_html = "<span class='status-pill status-listening'>🎙 Listening…</span>"
        if is_done:
            status_html = "<span class='status-pill status-idle'>Interview Finished</span>"

        tab_upd = gr.update()
        if is_done:
            # Wait for Sarah to finish speaking before switching to report
            await asyncio.sleep(estimated_dur + 1.0)
            tab_upd = gr.update(selected=2)
            streaming_stt_service.stop()

        progress = f"Round {min(num_ans + 1, target)} / {target}"
        
        yield (session_state, transcript, "📢 Speaking...", msg_text, audio_b64,
               progress, status_html, is_done, tab_upd, True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (session_state, "", f"Error: {e}", "", "", gr.update(),
               "<span class='status-pill status-idle'>Error</span>", False, gr.update(), False)


async def build_report(session_state: dict):
    """Build the final report card data (async for consistency)."""
    answers = session_state.get("answers", [])
    avg = session_state.get("total_score", 0.0)
    if not avg and answers:
        avg = sum(a["score"] for a in answers) / len(answers)

    rows = []
    questions = session_state.get("questions", [])
    for i, ans in enumerate(answers):
        q_text = "—"
        if i < len(questions):
            q = questions[i]
            q_text = q["text"] if isinstance(q, dict) else q.text
        rows.append([
            f"Q{i+1}",
            q_text[:60] + "…" if len(q_text) > 60 else q_text,
            ans.get("text", "")[:60] + "…" if len(ans.get("text", "")) > 60 else ans.get("text", ""),
            f"{ans.get('score', 0):.0%}",
            ans.get("feedback", ""),
        ])

    pct = int(avg * 100)
    grade = "A" if pct >= 85 else "B" if pct >= 70 else "C" if pct >= 55 else "D"
    emoji = "🏆" if pct >= 85 else "🎯" if pct >= 70 else "📈"
    summary = f"{emoji} Overall Score: **{pct}%** — Grade: **{grade}**"
    return summary, rows

# ─── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Mock Interview") as demo:

    session_state = gr.State(make_fresh_state())
    extracted_skills = gr.State([])
    interview_started = gr.State(False)

    # ── Header ──
    gr.HTML("""
    <div class="header-block">
        <h1>🤖 AI Mock Interview <span style="font-size:0.6em;opacity:0.5;">v1.1</span></h1>
        <p>Practice technical interviews with AI — powered by Azure OpenAI &amp; Azure Speech</p>
    </div>
    """)

    with gr.Tabs() as tabs:

        # ══════════════════════════════════════════════
        #  TAB 1 — Setup
        # ══════════════════════════════════════════════
        with gr.TabItem("📋 Setup", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="glass-card">
                        <h3 style="color:#a78bfa;margin-bottom:0.8rem;">Upload Resume</h3>
                        <p style="color:#64748b;font-size:0.85rem;">Supports PDF, DOCX, TXT</p>
                    </div>
                    """)
                    resume_file = gr.File(
                        label="Resume File",
                        file_types=[".pdf", ".docx", ".txt"],
                        type="filepath",
                    )
                    parse_btn = gr.Button("🔍 Parse Resume", elem_classes=["primary-btn"], variant="primary")

                with gr.Column(scale=2):
                    gr.HTML('<div class="glass-card"><h3 style="color:#a78bfa;">Resume Preview</h3></div>')
                    resume_preview = gr.Textbox(
                        label="Extracted Text (preview)",
                        lines=6,
                        interactive=False,
                        placeholder="Resume content will appear here…",
                    )
                    skills_display = gr.Textbox(
                        label="Detected Skills",
                        interactive=False,
                        placeholder="Skills will be highlighted here…",
                    )
                    
                    with gr.Row():
                        difficulty_radio = gr.Radio(
                            choices=["Junior", "Mid-Level", "Senior", "Expert"],
                            value="Mid-Level",
                            label="Interview Difficulty",
                            container=True
                        )
                        num_q_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Questions",
                            container=True
                        )

            with gr.Row():
                start_btn = gr.Button(
                    "🚀 Start Interview", elem_classes=["primary-btn"], variant="primary", scale=1
                )
                gr.HTML('<div style="color:#64748b;font-size:0.82rem;padding:0.6rem;">5 personalised questions · ~10 min · Voice interaction</div>')

        # ══════════════════════════════════════════════
        #  TAB 2 — Interview Room (Video Call UI)
        # ══════════════════════════════════════════════
        with gr.TabItem("🎙 Interview", id=1):
            # 1. Video Grid
            with gr.Row(elem_classes=["video-grid"]):
                with gr.Column(elem_classes=["video-container"]):
                    gr.Video(sources=["webcam"], label="User Feed", interactive=True)
                
                with gr.Column(elem_classes=["ai-avatar-container"]):
                    speaking_ring = gr.HTML('<div class="speaking-ring"></div>', visible=False)
                    gr.Image(
                        value="tech_ai_mascot_avatar_1774336888317.png", 
                        elem_classes=["ai-avatar-img"],
                        show_label=False,
                        container=False
                    )

            # 2. Control Bar
            with gr.Row(elem_classes=["control-bar"]):
                live_mode = gr.Checkbox(label="Live Mode", value=False)
                mute_btn = gr.Button("🔇", size="sm")
                cam_btn = gr.Button("📷", size="sm")
                stop_btn = gr.Button("📞", variant="stop", size="sm")

            # 3. Info & Interaction
            with gr.Row():
                with gr.Column(scale=1):
                    status_html = gr.HTML("<span class='status-pill status-idle'>Idle — Awaiting Start</span>")
                    progress_lbl = gr.Label(label="Round", value="—")
                    # Component to receive b64 and trigger JS playback
                    ai_audio_b64 = gr.Textbox(visible=False, interactive=False)

                with gr.Column(scale=3):
                    question_box = gr.Textbox(
                        label="Interviewer:",
                        placeholder="Greeting...",
                        interactive=False,
                        lines=2
                    )
                    with gr.Row():
                        mic_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy", # NumPy for direct streaming
                        label="System Microphone Active",
                        streaming=True,
                        visible=False
                    )
                    submit_btn = gr.Button("🚀 Send Answer (Manual)", variant="primary", elem_classes=["primary-btn"], visible=False)

            with gr.Row():
                transcript_box = gr.Textbox(label="Last Transcript", interactive=False)
                feedback_box = gr.Textbox(label="Feedback Snippet", interactive=False)

            # 4. Timer for Live Mode & Transcript polling (0.15s tick for responsiveness)
            auto_timer = gr.Timer(0.15, active=False)
            
            # mic continues in background via Azure native SDK

        # ══════════════════════════════════════════════
        #  TAB 3 — Report Card
        # ══════════════════════════════════════════════
        with gr.TabItem("📊 Report Card", id=2):
            gr.HTML("""
            <div class="glass-card" style="text-align:center;">
                <h2 style="color:#a78bfa;margin-bottom:0.5rem;">Interview Complete</h2>
                <p style="color:#64748b;">Your detailed performance report</p>
            </div>
            """)
            report_summary = gr.Markdown("*Complete the interview to see your report.*")
            report_table = gr.Dataframe(
                headers=["#", "Question", "Your Answer", "Score", "Feedback"],
                label="Detailed Breakdown",
                interactive=False,
                wrap=True,
            )
            retry_btn = gr.Button("🔄 Start New Interview", elem_classes=["secondary-btn"])

    # ─── Internal trigger for switching to report tab ───────────────────────
    switch_tab = gr.State(False)

    # ════════════════════════════════════════════════════════════════
    #   Event wiring
    # ════════════════════════════════════════════════════════════════

    # 1. Parse resume
    parse_btn.click(
        fn=process_resume,
        inputs=[resume_file],
        outputs=[resume_preview, skills_display, extracted_skills],
    )

    # 1. Audio Playback Handler (Global for all audio events)
    ai_audio_b64.change(
        fn=None,
        inputs=[ai_audio_b64],
        js=PLAY_AUDIO_JS
    )

    # 2. Start interview
    start_btn.click(
        fn=start_interview,
        inputs=[resume_file, extracted_skills, difficulty_radio, num_q_slider, session_state],
        outputs=[session_state, question_box, ai_audio_b64, progress_lbl, status_html, tabs, speaking_ring, auto_timer],
        js=None,
    )

    # ── Streaming Open Mic Logic ──
    # Removed browser stream wiring as we use system microphone directly in StreamingSTTService

    async def _check_transcript(state):
        """Check for fully recognized sentences from Azure with silence detection."""
        import time
        import re
        
        # --- ECHO SUPPRESSION & BARGE-IN ---
        curr_time = time.time()
        sarah_speaking_until = state.get("speaking_until", 0.0)
        # We use a slight lead-time to avoid cutting off the end of Sarah's audio
        is_sarah_speaking = curr_time < (sarah_speaking_until - 0.2)
        
        text = streaming_stt_service.get_latest_transcript()
        
        # If Sarah is speaking and we see text, check for barge-in
        if is_sarah_speaking and text:
            if len(text.split()) > 3:
                print(f"DEBUG: Barge-in detected: '{text}'")
                # We DON'T return/yield here, allowing the text to be buffered below
            else:
                # Likely echo or short noise. We don't want to stop Sarah for this,
                # but we also don't want to discard it yet.
                print(f"DEBUG: Potential echo/short answer during Sarah's speech: '{text}'")
                # Fall through to buffering
        elif is_sarah_speaking and not text:
            # No final transcript yet, just Sarah speaking.
            yield (state,) + (gr.update(),) * 10
            return
        curr_time = time.time()
        
        if text:
            # Standard buffering for longer answers
            pending_val = (state.get("pending_transcript", "") + " " + text).strip()
            state["pending_transcript"] = pending_val
            state["last_transcript_time"] = curr_time
            # Update the transcript box in the UI immediately
            yield (state, pending_val) + (gr.update(),) * 9
            return

        # --- Intermediate Results Check ---
        inter_text = streaming_stt_service.get_intermediate_transcript()
        if inter_text:
             pending = state.get("pending_transcript", "")
             display_text = (pending + " " + inter_text).strip()
             # Fast visual feedback: "I'm listening..." or similar
             status_html_val = f"<span class='status-pill status-listening'>🎙 {inter_text[:30]}...</span>"
             # IMPORTANT: Reset the silence timer so we don't process mid-sentence
             state["last_transcript_time"] = curr_time
             # Update transcript box with partial text
             yield (state, display_text, gr.update(), gr.update(), gr.update(), gr.update(), status_html_val) + (gr.update(),) * 4
             return

        # No new text, check if we have pending text that has 'settled'
        pending = state.get("pending_transcript", "")
        last_time = state.get("last_transcript_time", 0.0)
        
        # Silence threshold: 0.4s after the last detected word before processing.
        # The intermediate-transcript path resets last_transcript_time while the user
        # is actively speaking, so this only fires after a genuine pause.
        if pending and (curr_time - last_time > 0.4):
            print(f"DEBUG: Silence detected (0.4s). Processing pending transcript: '{pending}'")
            state["pending_transcript"] = "" # Clear pending buffer
            
            # Yield THINKING state
            yield (state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "<span class='status-pill status-thinking'>🤔 Sarah is thinking...</span>", gr.update(), gr.update(), True, gr.update())
            
            async for result in submit_answer(None, state, override_transcript=pending):
                (
                    new_state, transcript, status_msg, q_text, audio, 
                    progress, status_html_val, is_done, tab_upd, ring_vis
                ) = result
                state.update(new_state)
                timer_upd = gr.update(active=False) if is_done else gr.update()
                yield (state, transcript, status_msg, q_text, audio, progress, status_html_val, is_done, tab_upd, ring_vis, timer_upd)
            return
        
        # Match the 11 outputs
        # Set speaking_ring (10th) to False if idling
        yield (state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update())

    # Update auto_timer tick to match the 11 outputs
    auto_timer.tick(
        fn=_check_transcript,
        inputs=[session_state],
        outputs=[
            session_state, transcript_box, feedback_box, question_box, ai_audio_b64, 
            progress_lbl, status_html, switch_tab, tabs, speaking_ring, auto_timer
        ]
    )

    # ... rest of event wiring ...

    # Automatically clear the transcript box when the user starts a new recording or uploads a new file
    mic_input.change(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[transcript_box, feedback_box],
    )

    # 4. When interview done, populate report card
    async def _handle_report(done, state):
        if done:
            return await build_report(state)
        return gr.update(), gr.update()

    switch_tab.change(
        fn=_handle_report,
        inputs=[switch_tab, session_state],
        outputs=[report_summary, report_table],
    )


    # 5. Retry / End Call
    def _reset():
        fresh = make_fresh_state()
        streaming_stt_service.stop()
        return fresh, [], "", "", "", "—", "<span class='status-pill status-idle'>Idle</span>", "*Complete the interview to see your report.*", [], gr.Tabs(selected=0), False, gr.update(active=False)

    retry_btn.click(
        fn=_reset,
        inputs=[],
        outputs=[session_state, extracted_skills, question_box, transcript_box, ai_audio_b64, progress_lbl, status_html, report_summary, report_table, tabs, speaking_ring, auto_timer],
    )
    stop_btn.click(
        fn=_reset,
        inputs=[],
        outputs=[session_state, extracted_skills, question_box, transcript_box, ai_audio_b64, progress_lbl, status_html, report_summary, report_table, tabs, speaking_ring, auto_timer],
    )


if __name__ == "__main__":
    # Pre-flight check (attempt to find backend on startup - skipped in local mode)
    pass

    try:
        demo.launch(
            server_name="127.0.0.1",
            share=False,
            show_error=True,
            css=CUSTOM_CSS,
        )
    finally:
        print("🛑 Shutting down services...")
        try:
            streaming_stt_service.stop()
        except Exception as e:
            print(f"Error stopping STT: {e}")
