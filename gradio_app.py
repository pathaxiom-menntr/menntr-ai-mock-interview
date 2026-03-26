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
import tempfile
import json
from pathlib import Path

# ─── Windows Connection Fatigue Fix ──────────────────────────────────────────
if sys.platform == "win32":
    def silent_exception_handler(loop, context):
        msg = context.get("message")
        exception = context.get("exception")
        if "WinError 10054" in str(msg) or "WinError 10054" in str(exception) or isinstance(exception, ConnectionResetError):
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

# ─── Lazy imports (avoid crash when .env not yet filled) ──────────────────────
from app.services.resume_parser import resume_parser_service
from app.services.speech_to_text import stt_service
from app.services.text_to_speech import tts_service
from app.ai.langgraph_flow import app_graph

# ─── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #07071a !important;
    color: #e2e8f0 !important;
    min-height: 100vh;
}

/* ── Header ── */
.header-block {
    background: linear-gradient(135deg, #1a0d3d 0%, #0d1a3d 100%);
    border-bottom: 1px solid rgba(124,58,237,0.35);
    padding: 1.5rem 2rem;
    text-align: center;
}
.header-block h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.header-block p {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Gradio component overrides ── */
.gr-button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.primary-btn {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important;
    color: #fff !important;
    padding: 0.7rem 1.8rem !important;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(124,58,237,0.4) !important;
}
.secondary-btn {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #cbd5e1 !important;
}

/* ── Avatar pulsing ring ── */
.avatar-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
}
.avatar-ring {
    width: 96px; height: 96px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 2.5rem;
    box-shadow: 0 0 0 0 rgba(124,58,237,0.7);
    animation: pulse-ring 2.5s infinite;
}
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0   rgba(124,58,237,0.7); }
    70%  { box-shadow: 0 0 0 18px rgba(124,58,237,0.0); }
    100% { box-shadow: 0 0 0 0   rgba(124,58,237,0.0); }
}
.avatar-ring.idle {
    animation: none;
    box-shadow: 0 0 0 4px rgba(124,58,237,0.3);
}

/* ── Status pill ── */
.status-pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.status-thinking { background: rgba(250,204,21,0.15); color: #fbbf24; border: 1px solid rgba(250,204,21,0.3); }
.status-listening { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.status-idle { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }

/* ── Score badge ── */
.score-badge {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Tabs */
.tab-nav button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 500 !important;
}
.tab-nav button.selected {
    color: #a78bfa !important;
    border-bottom: 2px solid #7c3aed !important;
}

/* Text areas / inputs */
.gr-textbox textarea, .gr-textbox input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
label span { color: #94a3b8 !important; font-size: 0.82rem !important; }

/* File upload */
.gr-file { background: rgba(255,255,255,0.04) !important; border-radius: 12px !important; }

/* Dataframe */
.gr-dataframe table { background: rgba(255,255,255,0.03) !important; }
.gr-dataframe th { background: rgba(124,58,237,0.15) !important; color: #a78bfa !important; }
.gr-dataframe td { border-color: rgba(255,255,255,0.07) !important; color: #cbd5e1 !important; }
"""

# ─── State helpers ─────────────────────────────────────────────────────────────

def make_fresh_state():
    return {
        "user_id": "demo_user",
        "resume_text": "",
        "skills": [],
        "questions": [],
        "current_question": None,
        "answers": [],
        "tool_calls": [],
        "tool_outputs": [],
        "total_score": 0.0,
        "is_finished": False,
        "last_user_input": "",
        "next_node": "",
    }

# ─── Interview question generation (direct LLM call, no LangGraph ainvoke) ────

async def _generate_question(skills: list, previous_questions: list) -> str:
    """Ask the LLM to produce one interview question based on skills."""
    from app.ai.llm_client import llm_client
    skills_str = ", ".join(skills) if skills else "General"
    prev_str = "\n".join(f"- {q}" for q in previous_questions) if previous_questions else "None"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical interviewer. "
                "Generate ONE clear, specific interview question tailored to the candidate's skills. "
                "Do NOT number the question. Return only the question text, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Candidate skills: {skills_str}\n"
                f"Previous questions asked (do NOT repeat):\n{prev_str}\n\n"
                "Generate the next interview question:"
            ),
        },
    ]
    response = await llm_client.get_completion(messages=messages)
    return response.choices[0].message.content.strip()


# ─── Backend functions ─────────────────────────────────────────────────────────

async def process_resume(resume_file):
    """Parse uploaded resume and extract skills."""
    if resume_file is None:
        return "⚠️ No file uploaded.", "", []

    file_path = resume_file.name if hasattr(resume_file, "name") else str(resume_file)
    try:
        text = resume_parser_service.parse(file_path)
        if not text.strip():
            return "⚠️ Empty resume file.", "", []

        # Extract skills via LLM (MCP tool called directly for speed in UI)
        from app.ai.mcp_server import extract_skills
        raw = await extract_skills(text)
        try:
            skills = json.loads(raw)
        except Exception:
            skills = ["General"]

        preview = text[:400] + "…" if len(text) > 400 else text
        skills_str = " · ".join(f"🔹 {s}" for s in skills)
        return preview, skills_str, skills

    except Exception as e:
        return f"❌ Error parsing resume: {str(e)}", "", []


async def start_interview(skills: list, session_state: dict):
    """Generate the first question directly via LLM (no LangGraph ainvoke)."""
    print(f"DEBUG: start_interview called with skills: {skills}")
    sys.stdout.flush()
    resolved_skills = skills if skills else ["General"]

    try:
        question_text = await _generate_question(resolved_skills, [])
        if not question_text:
            print("DEBUG: start_interview failed to generate question")
            sys.stdout.flush()
            return session_state, "❌ Could not generate a question. Check your Azure OpenAI credentials.", None, "1/5", "<span class='status-pill status-idle'>Error</span>", gr.Tabs(selected=0)

        audio_path = tts_service.speak_text(question_text)

        new_state = make_fresh_state()
        new_state["skills"] = resolved_skills
        q = {"id": "1", "text": question_text, "skill": resolved_skills[0] if resolved_skills else "General", "difficulty": "Medium"}
        new_state["current_question"] = q
        new_state["questions"] = [q]

        print(f"DEBUG: start_interview successful, returning question and tab switch to 'interview'")
        sys.stdout.flush()
        return (
            new_state,
            question_text,
            audio_path,
            "Question 1 / 5",
            "<span class='status-pill status-listening'>🎙 Listening…</span>",
            gr.Tabs(selected=1),
        )
    except Exception as e:
        print(f"DEBUG: start_interview ERROR: {e}")
        import traceback; traceback.print_exc()
        sys.stdout.flush()
        return session_state, f"❌ Error: {str(e)}", None, "1/5", "<span class='status-pill status-idle'>Error</span>", gr.Tabs(selected=0)


async def submit_answer(audio_input, session_state: dict):
    """Transcribe audio → evaluate answer → generate next question, all via direct async calls."""
    print(f"DEBUG: submit_answer CALLED with audio_input type: {type(audio_input)}")
    print(f"DEBUG: audio_input value: {audio_input}")
    sys.stdout.flush()

    if audio_input is None:
        print("DEBUG: audio_input is None - NO AUDIO RECEIVED")
        sys.stdout.flush()
        return session_state, "⚠️ No audio recorded.", "Please record an answer.", None, None, "–", "<span class='status-pill status-idle'>Waiting…</span>", False, gr.Tabs()

    # Check file existence if it's a string path
    if isinstance(audio_input, str):
        if os.path.exists(audio_input):
            size = os.path.getsize(audio_input)
            print(f"DEBUG: Audio file exists at {audio_input}, size: {size} bytes")
        else:
            print(f"DEBUG: Audio file path provided but FILE NOT FOUND: {audio_input}")
        sys.stdout.flush()
        
        if not os.path.exists(audio_input):
            return session_state, "❌ Audio file not found on server.", "", None, "–", "<span class='status-pill status-idle'>File Error</span>", False, gr.Tabs()
    elif isinstance(audio_input, dict):
        print(f"DEBUG: audio_input is a DICT: {audio_input.keys()}")
        sys.stdout.flush()
        # Gradio sometimes sends a dict with 'name' (path)
        path = audio_input.get("name")
        if path:
            print(f"DEBUG: Extracted path from dict: {path}")
            sys.stdout.flush()
            audio_input = path
        else:
            return session_state, "❌ Unexpected audio data format.", "", None, "–", "<span class='status-pill status-idle'>Format Error</span>", False, gr.Tabs()

    # ── Step 1: STT ──
    print("DEBUG: Starting transcription...")
    sys.stdout.flush()
    transcript = stt_service.transcribe_audio(audio_input)
    print(f"DEBUG: Transcription complete. Result: {transcript[:50]}...")
    sys.stdout.flush()

    if not transcript or transcript.startswith("STT Error") or "Could not understand" in transcript:
        return session_state, transcript, "❗ Could not understand audio. Please speak clearly and try again.", None, None, "–", "<span class='status-pill status-listening'>🎙 Try Again</span>", False, gr.Tabs()

    # ── Step 2: Evaluate + next question ──
    state = dict(session_state)
    q = state.get("current_question")
    if not q:
        return session_state, transcript, "No active question found. Please start the interview first.", None, None, "–", "<span class='status-pill status-idle'>Idle</span>", False, gr.Tabs()

    try:
        from app.ai.mcp_server import evaluate_answer as mcp_evaluate
        q_text = q["text"] if isinstance(q, dict) else q.text
        q_id   = q["id"]   if isinstance(q, dict) else q.id

        # Evaluate answer
        raw = await mcp_evaluate(q_text, transcript)
        try:
            # Strip markdown code fences if present
            clean = raw.strip().strip("```json").strip("```").strip()
            eval_data = json.loads(clean)
        except Exception:
            eval_data = {"score": 0.5, "feedback": raw}

        score    = float(eval_data.get("score", 0.5))
        feedback = eval_data.get("feedback", "Good attempt.")

        # Record answer
        answers = list(state.get("answers", []))
        answers.append({"question_id": q_id, "text": transcript, "score": score, "feedback": feedback})
        state["answers"] = answers

        # Done after 5 questions
        if len(answers) >= 5:
            state["total_score"] = sum(a["score"] for a in answers) / len(answers)
            state["is_finished"] = True
            answer_count = len(answers)
            print("DEBUG: submit_answer FINISHED, switching to 'report' tab")
            sys.stdout.flush()
            return (
                state,
                transcript,
                f"✅ {feedback}",
                gr.update(), # Keep current question text
                None,
                f"Done — {answer_count}/5 answers",
                "<span class='status-pill status-idle'>Interview Complete 🎉</span>",
                True,
                gr.Tabs(selected=2),
            )

        # Generate next question
        prev_questions = [q2["text"] if isinstance(q2, dict) else q2.text for q2 in state.get("questions", [])]
        next_q_text = await _generate_question(state.get("skills", ["General"]), prev_questions)
        next_q_num  = len(answers) + 1
        next_q = {"id": str(next_q_num), "text": next_q_text, "skill": "General", "difficulty": "Medium"}
        state["current_question"] = next_q
        state["questions"] = list(state.get("questions", [])) + [next_q]

        audio_path = tts_service.speak_text(next_q_text)
        answer_count = len(answers)
        progress   = f"Question {answer_count + 1} / 5"
        
        # Resilience delay for Windows socket flushing
        await asyncio.sleep(0.1)

        return (
            state,
            transcript,
            f"✅ Score {score:.0%} — {feedback}",
            next_q_text,
            audio_path,
            progress,
            "<span class='status-pill status-listening'>🎙 Listening…</span>",
            False,
            gr.update(),
        )
    except Exception as e:
        print(f"DEBUG: submit_answer ERROR: {e}")
        import traceback; traceback.print_exc()
        sys.stdout.flush()
        return session_state, f"Error: {e}", str(e), gr.update(), None, "–", "<span class='status-pill status-idle'>Error</span>", False, gr.Tabs()


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

with gr.Blocks(css=CUSTOM_CSS, title="AI Mock Interview") as demo:

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
                start_btn = gr.Button(
                    "🚀 Start Interview", elem_classes=["primary-btn"], variant="primary", scale=1
                )
                gr.HTML('<div style="color:#64748b;font-size:0.82rem;padding:0.6rem;">5 personalised questions · ~10 min · Voice interaction</div>')

        # ══════════════════════════════════════════════
        #  TAB 2 — Interview Room
        # ══════════════════════════════════════════════
        with gr.TabItem("🎙 Interview", id=1):
            with gr.Row():
                # Left: AI avatar + question
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="glass-card avatar-wrap">
                        <div class="avatar-ring">🤖</div>
                        <span style="color:#a78bfa;font-weight:600;font-size:1rem;">AI Interviewer</span>
                    </div>
                    """)
                    status_html = gr.HTML("<span class='status-pill status-idle'>Idle — Start interview to begin</span>")
                    progress_lbl = gr.Label(label="Progress", value="—")

                # Middle: Question + AI audio
                with gr.Column(scale=2):
                    gr.HTML('<div class="glass-card"><h3 style="color:#a78bfa;margin-bottom:0.5rem;">💬 Question</h3></div>')
                    question_box = gr.Textbox(
                        label="",
                        lines=5,
                        interactive=False,
                        placeholder="The AI interviewer's question will appear here…",
                    )
                    ai_audio_out = gr.Audio(
                        label="🔊 AI Interviewer (auto-play)",
                        autoplay=True,
                        interactive=False,
                        type="filepath",
                    )

                # Right: Mic + transcript + feedback
                with gr.Column(scale=2):
                    gr.HTML('<div class="glass-card"><h3 style="color:#a78bfa;margin-bottom:0.5rem;">🎤 Your Answer</h3></div>')
                    mic_input = gr.Audio(
                        sources=["microphone", "upload"],
                        label="Record or Upload your answer",
                        type="filepath",
                        # Removed format="wav" to avoid ffmpeg dependency on Windows
                        interactive=True,
                    )
                    submit_btn = gr.Button("✅ Submit Answer", elem_classes=["primary-btn"], variant="primary")

                    transcript_box = gr.Textbox(
                        label="📝 Transcribed Answer",
                        lines=3,
                        interactive=False,
                        placeholder="Your words will appear here after submission…",
                    )
                    feedback_box = gr.Textbox(
                        label="💡 AI Feedback",
                        lines=4,
                        interactive=False,
                        placeholder="Score and feedback appear here…",
                    )

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

    # 2. Start interview
    start_btn.click(
        fn=start_interview,
        inputs=[extracted_skills, session_state],
        outputs=[session_state, question_box, ai_audio_out, progress_lbl, status_html, tabs],
    )

    # 3. Submit answer
    submit_btn.click(
        fn=submit_answer,
        inputs=[mic_input, session_state],
        outputs=[session_state, transcript_box, feedback_box, question_box, ai_audio_out, progress_lbl, status_html, switch_tab, tabs],
    )

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


    # 5. Retry
    def _reset():
        print("DEBUG: _reset called, switching to 'setup' tab")
        sys.stdout.flush()
        fresh = make_fresh_state()
        return fresh, [], "", "", None, "—", "<span class='status-pill status-idle'>Idle</span>", "*Complete the interview to see your report.*", [], gr.Tabs(selected=0)

    retry_btn.click(
        fn=_reset,
        inputs=[],
        outputs=[session_state, extracted_skills, question_box, transcript_box, ai_audio_out, progress_lbl, status_html, report_summary, report_table, tabs],
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            loop = asyncio.get_event_loop()
            def silent_handler(loop, context):
                msg = context.get("message")
                exc = context.get("exception")
                if "WinError 10054" in str(msg) or "WinError 10054" in str(exc) or isinstance(exc, ConnectionResetError):
                    return
                loop.default_exception_handler(context)
            loop.set_exception_handler(silent_handler)
        except Exception:
            pass

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
