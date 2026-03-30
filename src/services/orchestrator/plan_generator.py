"""Interview Plan Generator.

Runs ONCE before the greeting. Analyzes resume + job description to produce:
  - A seniority estimate (junior/mid/senior/staff_principal)
  - An ordered list of TopicPlan objects (what to cover, in what order, with what depth)
  - A flag indicating whether a coding assessment is needed

Why this exists (vs the old reactive approach):
  Old system: LLM invents a question every turn with no blueprint — leads to random
  topic jumps, missed coverage, and uniform depth regardless of candidate level.

  New system: We build a structured plan upfront. The question_node picks from the plan.
  The depth engine (in control_nodes) decides when a topic is "done" based on seniority
  rules, so a Staff Engineer gets pushed much harder than a junior candidate.
"""

import logging
import json
import uuid
from typing import Optional

from src.services.orchestrator.llm_helpers import LLMHelper
from src.services.orchestrator.context_builders import build_resume_context, build_job_context
from src.services.orchestrator.constants import (
    SENIORITY_JUNIOR, SENIORITY_MID, SENIORITY_SENIOR, SENIORITY_STAFF,
    SENIORITY_LEVELS, DEPTH_RULES,
    TOPIC_BACKGROUND, TOPIC_TECHNICAL, TOPIC_BEHAVIORAL,
    TOPIC_SITUATIONAL, TOPIC_PROJECT, TOPIC_CODING, TOPIC_CATEGORIES,
    COVERAGE_PENDING, COVERAGE_IN_PROGRESS, PRIORITY_MUST_ASK, PRIORITY_SHOULD_ASK, PRIORITY_NICE_TO_HAVE,
    STYLE_TECHNICAL_HEAVY, STYLE_BEHAVIORAL_HEAVY, STYLE_BALANCED,
    TEMPERATURE_ANALYTICAL, TEMPERATURE_BALANCED,
)

logger = logging.getLogger(__name__)


async def generate_interview_plan(state: dict, llm_helper: LLMHelper) -> dict:
    """Generate a structured interview plan from resume + job description.

    Returns a dict (InterviewPlan) with:
        topics: list[TopicPlan]
        seniority_level: str
        expected_depth: str
        requires_coding: bool
        coding_language: str | None
        target_turns: int
        interview_style: str
    """
    resume_context = build_resume_context(state)
    job_context = build_job_context(state)

    # Step 1: Estimate candidate seniority from resume + JD
    seniority = await _estimate_seniority(resume_context, job_context, llm_helper)

    # Step 2: Generate topic list calibrated to seniority
    topics = await _generate_topics(resume_context, job_context, seniority, llm_helper)

    # Step 3: Derive plan-level metadata
    requires_coding = any(t["category"] == TOPIC_CODING for t in topics)
    coding_language = _detect_primary_language(resume_context, job_context)

    depth_rules = DEPTH_RULES[seniority]

    technical_count = sum(
        1 for t in topics
        if t["category"] in [TOPIC_TECHNICAL, TOPIC_CODING, TOPIC_PROJECT]
    )
    behavioral_count = sum(
        1 for t in topics
        if t["category"] in [TOPIC_BEHAVIORAL, TOPIC_SITUATIONAL]
    )
    if technical_count > behavioral_count * 1.5:
        interview_style = STYLE_TECHNICAL_HEAVY
    elif behavioral_count > technical_count * 1.5:
        interview_style = STYLE_BEHAVIORAL_HEAVY
    else:
        interview_style = STYLE_BALANCED

    # Estimate total turns: each topic takes 1 + (max_iterations - 1) turns on average
    must_ask_topics = [t for t in topics if t["priority"] <= PRIORITY_SHOULD_ASK]
    target_turns = sum(
        t["max_iterations"]
        for t in must_ask_topics
    )
    target_turns = max(8, min(target_turns, 25))

    plan = {
        "topics": topics,
        "seniority_level": seniority,
        "expected_depth": depth_rules["expected_depth"],
        "requires_coding": requires_coding,
        "coding_language": coding_language,
        "target_turns": target_turns,
        "interview_style": interview_style,
    }

    logger.info(
        f"Interview plan generated: seniority={seniority}, topics={len(topics)}, "
        f"requires_coding={requires_coding}, style={interview_style}, target_turns={target_turns}"
    )
    return plan


async def _estimate_seniority(
    resume_context: str,
    job_context: str,
    llm_helper: LLMHelper,
) -> str:
    """Estimate candidate seniority level from resume and JD."""

    prompt = f"""Analyze the resume and job description to estimate the candidate's seniority level.

RESUME:
{resume_context or "No resume provided."}

JOB DESCRIPTION:
{job_context or "No job description provided."}

Seniority definitions:
- "junior": 0-2 years experience. Entry-level titles: intern, associate, junior dev, graduate.
- "mid": 2-5 years. Standard contributor: software engineer, developer, SDE-II.
- "senior": 5-9 years. Senior titles: senior engineer, tech lead, SDE-III.
- "staff_principal": 9+ years. Leadership: staff engineer, principal, architect, director.

Evidence to weigh (in order of importance):
1. Years of experience explicitly stated
2. Progression of job titles
3. Complexity and scope of described projects
4. JD's required years of experience (if stated)
5. Leadership or mentorship signals

Return JSON only: {{"seniority": "junior|mid|senior|staff_principal", "reasoning": "1-sentence justification"}}"""

    try:
        result_json = await llm_helper.call_llm_json(
            system_prompt="You are an expert at evaluating engineering seniority. Be evidence-based and precise. Return only valid JSON.",
            user_prompt=prompt,
            temperature=TEMPERATURE_ANALYTICAL,
        )
        result = json.loads(result_json)
        seniority = result.get("seniority", SENIORITY_MID)
        if seniority not in SENIORITY_LEVELS:
            seniority = SENIORITY_MID
        logger.info(f"Seniority estimated: {seniority} — {result.get('reasoning', '')}")
        return seniority
    except Exception as e:
        logger.warning(f"Seniority estimation failed ({e}), defaulting to mid")
        return SENIORITY_MID


async def _generate_topics(
    resume_context: str,
    job_context: str,
    seniority: str,
    llm_helper: LLMHelper,
) -> list[dict]:
    """Generate an ordered, depth-calibrated topic plan for the interview."""

    depth_rules = DEPTH_RULES[seniority]

    prompt = f"""You are an expert technical interviewer. Create a structured interview plan.

RESUME:
{resume_context or "No resume provided."}

JOB DESCRIPTION:
{job_context or "No job description provided."}

CANDIDATE SENIORITY: {seniority}
EXPECTED DEPTH: {depth_rules["expected_depth"]}
PROBE STYLE: {depth_rules["probe_style"]}

Generate 6-10 topics that cover the interview comprehensively. Order them:
  1. Warm-up / background (1 topic, priority 1)
  2. Technical deep-dives from resume skills/projects (2-4 topics, priority 1-2)
  3. Behavioral STAR questions (1-2 topics, priority 1-2)
  4. Situational / hypothetical (1-2 topics, priority 2-3)
  5. Coding assessment — ONLY if JD requires coding OR seniority is senior/staff_principal (1 topic, priority 1, category "coding")

For EACH topic, set iteration limits based on seniority:
  - background:    max_iterations = {depth_rules["behavioral_max_iterations"]}
  - technical:     max_iterations = {depth_rules["technical_max_iterations"]}
  - project:       max_iterations = {depth_rules["technical_max_iterations"]}
  - behavioral:    max_iterations = {depth_rules["behavioral_max_iterations"]}
  - situational:   max_iterations = 1
  - coding:        max_iterations = 1

Quality threshold to advance: {depth_rules["min_quality_to_advance"]}

Rules for good topics:
  - Be SPECIFIC (not "Python skills" but "Python async/await and the event loop")
  - Anchor to actual resume content where possible
  - initial_question must be conversational and open-ended, not a quiz question
  - Coding topics should NOT show the code editor until the agent explicitly triggers it

Return a JSON array (no wrapping object):
[
  {{
    "topic": "specific topic name",
    "category": "background|technical|behavioral|situational|project|coding",
    "priority": 1|2|3,
    "source": "resume_project|jd_requirement|standard_behavioral|standard_technical",
    "initial_question": "opening question for this topic",
    "max_iterations": 1|2|3,
    "min_quality_to_advance": 0.0-1.0,
    "requires_code": true|false
  }}
]"""

    try:
        result_json = await llm_helper.call_llm_json(
            system_prompt=(
                "You are an expert technical interviewer creating a structured plan. "
                "Return ONLY a valid JSON array, no markdown, no extra text."
            ),
            user_prompt=prompt,
            temperature=TEMPERATURE_BALANCED,
        )

        raw = json.loads(result_json)
        # Handle both bare array and wrapped {"topics": [...]}
        if isinstance(raw, dict):
            raw = raw.get("topics", raw.get("topic_plan", []))

        topics: list[dict] = []
        for item in raw:
            category = item.get("category", TOPIC_BACKGROUND)
            if category not in TOPIC_CATEGORIES:
                category = TOPIC_BACKGROUND

            topic: dict = {
                "id": str(uuid.uuid4()),
                "topic": str(item.get("topic", "General background")),
                "category": category,
                "priority": int(item.get("priority", PRIORITY_SHOULD_ASK)),
                "source": str(item.get("source", "standard")),
                "initial_question": str(item.get("initial_question", "")),
                "max_iterations": min(int(item.get("max_iterations", 2)), 3),
                "min_quality_to_advance": float(
                    max(min(item.get("min_quality_to_advance", depth_rules["min_quality_to_advance"]), 1.0), 0.0)
                ),
                "requires_code": bool(item.get("requires_code", False)),
                # Runtime tracking fields
                "coverage_status": COVERAGE_PENDING,
                "iterations_done": 0,
                "last_quality_score": None,
            }
            topics.append(topic)

        # Sort by priority so must-ask topics come first
        topics.sort(key=lambda x: x["priority"])

        if topics:
            return topics

    except Exception as e:
        logger.warning(f"Topic generation failed ({e}), using fallback topics")

    return _fallback_topics(seniority)


def _fallback_topics(seniority: str) -> list[dict]:
    """Minimal fallback topic list when LLM generation fails."""
    d = DEPTH_RULES[seniority]
    return [
        {
            "id": str(uuid.uuid4()),
            "topic": "Career background and motivation",
            "category": TOPIC_BACKGROUND,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard",
            "initial_question": "Tell me a bit about yourself and what led you to apply for this role.",
            "max_iterations": 1,
            "min_quality_to_advance": d["min_quality_to_advance"],
            "requires_code": False,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
        },
        {
            "id": str(uuid.uuid4()),
            "topic": "Most challenging technical project",
            "category": TOPIC_PROJECT,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard_technical",
            "initial_question": "Walk me through the most technically challenging project you have worked on.",
            "max_iterations": d["technical_max_iterations"],
            "min_quality_to_advance": d["min_quality_to_advance"],
            "requires_code": False,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
        },
        {
            "id": str(uuid.uuid4()),
            "topic": "Problem solving under uncertainty",
            "category": TOPIC_BEHAVIORAL,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard_behavioral",
            "initial_question": "Tell me about a time you had to make a difficult decision with incomplete information.",
            "max_iterations": d["behavioral_max_iterations"],
            "min_quality_to_advance": d["min_quality_to_advance"],
            "requires_code": False,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
        },
    ]


def _detect_primary_language(resume_context: str, job_context: str) -> Optional[str]:
    """Detect the primary programming language from resume and job description text."""
    text = (resume_context + " " + job_context).lower()

    # Each language mapped to strong signal keywords
    language_signals: dict[str, list[str]] = {
        "python": ["python", "django", "flask", "fastapi", "pandas", "pytorch", "tensorflow"],
        "javascript": ["javascript", "typescript", "node.js", "nodejs", "react", "vue", "angular", "next.js"],
        "java": ["java", "spring", "maven", "gradle", "kotlin", "jvm"],
        "go": ["golang", " go ", "goroutine", "go modules"],
        "rust": ["rust ", "cargo", "rustlang"],
        "c++": ["c++", "cpp", "c plus plus", "stl", "boost"],
    }

    scores: dict[str, int] = {}
    for lang, signals in language_signals.items():
        scores[lang] = sum(text.count(s) for s in signals)

    if not any(scores.values()):
        return None

    return max(scores, key=lambda k: scores[k])


def get_next_pending_topic(plan: dict) -> Optional[dict]:
    """Return the next topic that hasn't been adequately covered yet."""
    topics = plan.get("topics", [])
    for topic in topics:
        if topic.get("coverage_status") in (COVERAGE_PENDING, COVERAGE_IN_PROGRESS):
            return topic
    return None


def get_topic_by_id(plan: dict, topic_id: str) -> Optional[dict]:
    """Find a topic in the plan by its ID."""
    for topic in plan.get("topics", []):
        if topic.get("id") == topic_id:
            return topic
    return None


def update_topic_in_plan(plan: dict, topic_id: str, updates: dict) -> dict:
    """Return a new plan dict with the specified topic updated (no mutation)."""
    new_topics = []
    for topic in plan.get("topics", []):
        if topic.get("id") == topic_id:
            new_topics.append({**topic, **updates})
        else:
            new_topics.append(topic)
    return {**plan, "topics": new_topics}
