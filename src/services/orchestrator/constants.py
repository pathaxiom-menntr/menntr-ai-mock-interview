"""Constants for interview orchestrator nodes.

ALL configuration values live here. No magic numbers scattered across files.
"""

# ============================================================================
# BASE SYSTEM PROMPT
# ============================================================================

COMMON_SYSTEM_PROMPT = """You are an authentic interviewer having a natural conversation. Your responses will be spoken aloud.

Core principles:
- ALWAYS respond in English only, regardless of what language the candidate uses
- Be authentic and genuine - not formulaic or robotic
- Be natural and conversational - not sycophantic or overly enthusiastic
- You have full context of the conversation, resume, and job requirements
- Trust your judgment and adapt to the conversation flow
- Use shorter sentences. Break up long thoughts. Speak like a real person, not a formal document.
- Vary your sentence length. Mix short and medium sentences for natural flow.
- Be direct and clear. Avoid unnecessary words or overly complex phrasing.
- If you know the candidate's name, use it naturally and appropriately - it makes the conversation more personal

Format for speech:
- Avoid colons (use periods or commas instead)
- Use commas instead of em dashes
- Write percentages as '5 percent' not '5%'
- Ensure sentences end with proper punctuation
- Keep sentences under 20 words when possible. Use pauses (commas) instead of long sentences."""

# ============================================================================
# INTERVIEW TERMINATION
# ============================================================================

# Exact message to output when candidate is rude/inappropriate. Do not soften.
TERMINATION_MESSAGE = (
    "Your behavior, attitude, and manner are inappropriate and intolerable "
    "in a professional setting. This interview is now terminated."
)

# Phrase used when returning to interview after answering a candidate's question
STANDARD_TRANSITION = "Does that help? Now, going back to what I was asking"

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE_CREATIVE = 0.8       # Greetings, conversational responses
TEMPERATURE_BALANCED = 0.7       # Decisions, persona generation
TEMPERATURE_ANALYTICAL = 0.3     # Analysis, plan generation, scoring
TEMPERATURE_QUESTION = 0.85      # Question generation (slightly more creative)

# ============================================================================
# SENIORITY LEVELS
# ============================================================================

SENIORITY_JUNIOR = "junior"           # 0–2 years, entry-level
SENIORITY_MID = "mid"                 # 2–5 years, standard contributor
SENIORITY_SENIOR = "senior"           # 5–9 years, senior/tech lead
SENIORITY_STAFF = "staff_principal"   # 9+ years, staff/principal/architect

SENIORITY_LEVELS = [SENIORITY_JUNIOR, SENIORITY_MID, SENIORITY_SENIOR, SENIORITY_STAFF]

# ============================================================================
# DEPTH ENGINE RULES
# Per seniority: how hard to probe, how many follow-ups, quality bar to advance
# ============================================================================

DEPTH_RULES: dict[str, dict] = {
    SENIORITY_JUNIOR: {
        # Max follow-ups before moving to next topic
        "conceptual_max_iterations": 1,
        "technical_max_iterations": 2,
        "behavioral_max_iterations": 1,
        # Quality score (0–1) the answer must reach before the depth engine moves on
        "min_quality_to_advance": 0.40,
        # Descriptor used in prompts to calibrate question complexity
        "expected_depth": "foundational",
        # Tone of follow-up probes
        "probe_style": "exploratory and supportive",
    },
    SENIORITY_MID: {
        "conceptual_max_iterations": 1,
        "technical_max_iterations": 2,
        "behavioral_max_iterations": 2,
        "min_quality_to_advance": 0.55,
        "expected_depth": "applied",
        "probe_style": "probing and specific",
    },
    SENIORITY_SENIOR: {
        "conceptual_max_iterations": 2,
        "technical_max_iterations": 3,
        "behavioral_max_iterations": 2,
        "min_quality_to_advance": 0.65,
        "expected_depth": "expert",
        "probe_style": "challenging and trade-off focused",
    },
    SENIORITY_STAFF: {
        "conceptual_max_iterations": 2,
        "technical_max_iterations": 3,
        "behavioral_max_iterations": 3,
        "min_quality_to_advance": 0.75,
        "expected_depth": "architect",
        "probe_style": "systems-thinking and cross-team impact focused",
    },
}

# ============================================================================
# TOPIC CATEGORIES
# ============================================================================

TOPIC_BACKGROUND = "background"      # Career history, motivations
TOPIC_TECHNICAL = "technical"        # Skills, tools, frameworks
TOPIC_BEHAVIORAL = "behavioral"      # STAR-method: past behavior
TOPIC_SITUATIONAL = "situational"    # Hypothetical scenarios
TOPIC_PROJECT = "project"            # Specific project deep-dives
TOPIC_CODING = "coding"              # Live coding assessment

TOPIC_CATEGORIES = [
    TOPIC_BACKGROUND, TOPIC_TECHNICAL, TOPIC_BEHAVIORAL,
    TOPIC_SITUATIONAL, TOPIC_PROJECT, TOPIC_CODING,
]

# ============================================================================
# TOPIC COVERAGE STATUS
# ============================================================================

COVERAGE_PENDING = "pending"           # Not yet discussed
COVERAGE_IN_PROGRESS = "in_progress"   # Currently being probed
COVERAGE_ADEQUATE = "adequate"         # Sufficient information gathered
COVERAGE_SKIPPED = "skipped"           # Skipped (time / not relevant)

# ============================================================================
# TOPIC PRIORITIES
# ============================================================================

PRIORITY_MUST_ASK = 1      # Core to this role — must be covered
PRIORITY_SHOULD_ASK = 2    # Important — ask unless time is short
PRIORITY_NICE_TO_HAVE = 3  # Interesting but optional

# ============================================================================
# INTERVIEW STYLES
# ============================================================================

STYLE_TECHNICAL_HEAVY = "technical_heavy"     # Mostly technical + coding
STYLE_BEHAVIORAL_HEAVY = "behavioral_heavy"   # Mostly behavioral + situational
STYLE_BALANCED = "balanced"                   # Mix of both

# ============================================================================
# INTERVIEW FLOW THRESHOLDS
# ============================================================================

SUMMARY_UPDATE_INTERVAL = 5                   # Update summary every N turns
MAX_CONVERSATION_LENGTH_FOR_SUMMARY = 30      # Also update if history exceeds this

# Guard rails for interview length
MIN_TURNS_BEFORE_CLOSING = 6                  # Don't close before this many turns
MAX_TURNS_BEFORE_EVALUATION = 30              # Force evaluation after this many turns

# ============================================================================
# PROBE FOLLOW-UP STYLE DESCRIPTORS (fed into prompts per seniority)
# ============================================================================

PROBE_DEPTH_DESCRIPTORS: dict[str, list[str]] = {
    "foundational": [
        "walk me through that a bit more",
        "can you give me a specific example of that",
        "how did that work in practice",
    ],
    "applied": [
        "could you go deeper into the technical details",
        "how did you handle the trade-offs there",
        "what would you do differently approaching it today",
    ],
    "expert": [
        "how would that solution hold up under heavy load",
        "what were the architectural trade-offs you weighed",
        "how did you validate or benchmark that approach",
    ],
    "architect": [
        "how does that decision fit into the broader system design",
        "what failure modes did you plan for",
        "how would you evolve this over the next couple of years",
    ],
}

# ============================================================================
# INAPPROPRIATE CONTENT — FAST PRE-CHECK
# These patterns are checked BEFORE the LLM intent detection as a safety net.
# The LLM handles subtle cases; this list catches explicit content reliably.
# ============================================================================

INAPPROPRIATE_PATTERNS: list[str] = [
    "have sex", "want sex", "sex with you", "fuck you", "fuck off", "motherfucker",
    "suck my", "blow me", "jerk off", "masturbat", "rape", "molest",
    "i'll kill", "i will kill", "gonna kill", "death threat",
    "racist", "nigger", "faggot", "retard",
]

# ============================================================================
# SANDBOX MONITORING
# ============================================================================

SANDBOX_POLL_INTERVAL_SECONDS = 10.0
SANDBOX_STUCK_THRESHOLD_SECONDS = 30.0
