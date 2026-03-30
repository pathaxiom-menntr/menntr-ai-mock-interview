"""Type definitions for the interview orchestrator."""

import operator
from typing import TypedDict, Literal, Optional, Annotated
from pydantic import BaseModel, Field


# ============================================================================
# STATE TYPES
# ============================================================================

class TopicPlan(TypedDict):
    """A single topic to cover in the interview, with depth configuration."""
    id: str                        # Unique ID for this topic
    topic: str                     # e.g. "Python async/await and event loop"
    category: str                  # background | technical | behavioral | situational | project | coding
    priority: int                  # 1=must_ask, 2=should_ask, 3=nice_to_have
    source: str                    # resume_project | jd_requirement | standard_behavioral | standard_technical
    initial_question: str          # Suggested opening question for this topic
    max_iterations: int            # Max follow-up probes before moving on (1–3)
    min_quality_to_advance: float  # Quality score threshold to move to next topic
    requires_code: bool            # Whether this topic should trigger the code editor
    # Runtime fields (updated during interview)
    coverage_status: str           # pending | in_progress | adequate | skipped
    iterations_done: int           # How many probes have been made on this topic
    last_quality_score: Optional[float]  # Last answer quality score for this topic


class QuestionRecord(TypedDict):
    """Record of a question asked during the interview."""
    id: str
    text: str
    source: str              # resume | followup | plan | user_request
    resume_anchor: Optional[str]
    aspect: str
    asked_at_turn: int
    planned_topic_id: Optional[str]  # Links to TopicPlan.id


class UserIntent(TypedDict):
    """Detected user intent from their response."""
    type: str  # write_code | review_code | technical_assessment | change_topic | clarify | candidate_question | stop | continue | rude_or_inappropriate | no_intent
    confidence: float
    extracted_from: str
    turn: int
    metadata: Optional[dict]


class SandboxState(TypedDict):
    """State of code sandbox activity."""
    is_active: bool
    last_activity_ts: float
    submissions: list[dict]
    signals: list[str]
    initial_code: str
    exercise_description: str
    exercise_difficulty: str
    exercise_hints: list[str]
    last_code_snapshot: str
    last_poll_time: float


class InterviewState(TypedDict):
    """Robust state schema for LangGraph interview workflow with reducers.

    Fields annotated with operator.add use LangGraph reducers for append-only operations.
    """
    # Core identifiers
    interview_id: int
    user_id: int
    resume_id: int | None
    candidate_name: str | None

    # Conversation - APPEND ONLY (uses reducer)
    turn_count: int
    conversation_history: Annotated[list[dict], operator.add]

    # Questions tracking - APPEND ONLY (uses reducer)
    questions_asked: Annotated[list[QuestionRecord], operator.add]
    current_question: str | None

    # Resume understanding
    resume_structured: dict
    topics_covered: list[str]

    # Job context
    job_description: str | None

    # ── NEW: Interview Plan ──────────────────────────────────────────────────
    # Generated once before the greeting, stored for the whole interview.
    # Contains TopicPlan list, seniority, depth rules, coding flag, etc.
    interview_plan: dict | None          # Serialized InterviewPlan dict

    # Which plan topic is currently being probed (TopicPlan.id)
    current_topic_id: str | None

    # Per-topic runtime tracking: {topic_id: {iterations_done, last_quality_score}}
    # NOT a reducer — updated in-place by finalize_turn_node
    topic_iterations: dict

    # Estimated seniority level (set by plan_interview_node)
    seniority_level: str | None          # junior | mid | senior | staff_principal

    # Signal to frontend: show the code editor panel
    show_code_editor: bool
    # ────────────────────────────────────────────────────────────────────────

    # User intent - APPEND ONLY (uses reducer)
    detected_intents: Annotated[list[UserIntent], operator.add]
    active_user_request: UserIntent | None

    # Sandbox / code
    sandbox: SandboxState

    # Flow control
    phase: str
    last_node: str
    next_node: str | None

    # Runtime fields
    answer_quality: float
    next_message: str | None
    last_response: str | None
    current_code: str | None
    code_execution_result: dict | None
    code_quality: dict | None
    code_submissions: Annotated[list[dict], operator.add]
    feedback: dict | None

    # Conversation summary (for memory management)
    conversation_summary: str

    # System
    checkpoints: Annotated[list[str], operator.add]


# ============================================================================
# PYDANTIC MODELS FOR LLM INTEGRATION
# ============================================================================

class UserIntentDetection(BaseModel):
    """LLM-driven user intent detection."""
    intent_type: Literal[
        "write_code", "review_code", "technical_assessment", "change_topic",
        "clarify", "candidate_question", "stop", "continue",
        "rude_or_inappropriate", "no_intent"
    ] = Field(..., description="Type of user intent")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this intent was detected")
    metadata: dict = Field(default_factory=dict)


class NextActionDecision(BaseModel):
    """LLM-driven decision on what to do next."""
    action: Literal[
        "greeting", "question", "followup", "closing",
        "evaluation", "sandbox_guidance", "code_review",
        "answer_candidate_question", "termination"
    ] = Field(..., description="What action to take next")
    reasoning: str = Field(..., description="Brief reasoning for this decision")


class QuestionGeneration(BaseModel):
    """Generated question with metadata."""
    question: str = Field(..., description="The question text")
    resume_anchor: Optional[str] = Field(None)
    aspect: str = Field(..., description="What aspect we're exploring")
    reasoning: str = Field(..., description="Why this question was chosen")
