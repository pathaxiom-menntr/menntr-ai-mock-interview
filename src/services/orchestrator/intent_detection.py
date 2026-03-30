"""User intent detection for interview orchestrator."""

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from src.services.orchestrator.types import InterviewState, UserIntent, UserIntentDetection
from src.services.orchestrator.context_builders import build_conversation_context
from src.services.orchestrator.llm_helpers import LLMHelper

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.services.logging.interview_logger import InterviewLogger


async def detect_user_intent(
    state: InterviewState,
    openai_client: AsyncOpenAI,
    interview_logger=None
) -> InterviewState:
    """Detect user intent from their last response.

    Returns state updates (no mutations) following LangGraph principles.
    """
    last_response = state.get("last_response")
    if not last_response:
        return {
            "active_user_request": None,
        }

    # Build conversation context - use last 10 messages instead of character limit
    conversation_history = state.get("conversation_history", [])
    recent_messages = conversation_history[-10:] if len(
        conversation_history) > 10 else conversation_history
    recent_context = "\n".join([
        f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
        for msg in recent_messages
    ])

    # Get the last question asked
    last_question = ""
    for msg in reversed(recent_messages):
        if msg.get("role") == "assistant":
            last_question = msg.get("content", "")
            break

    prompt = f"""Analyze the user's response to identify their TRUE INTENT. Focus on their GOAL, not keywords.

CONVERSATION (last 10 messages):
{recent_context}

LAST QUESTION: {last_question if last_question else "None (initial greeting)"}
USER RESPONSE: "{last_response}"

INTENT TYPES (choose the user's GOAL):

1. **write_code** - User wants to CREATE/WRITE code to demonstrate skills
   Examples: "I'd like to write code", "Can I show you my approach?", "Let me code something"
   NOT: Just mentioning code in an answer

2. **review_code** - User wants to SHARE/DISCUSS existing code
   Examples: "Here's my code", "Can you review this?", "I have code to show"
   NOT: Asking to write new code

3. **change_topic** - User wants to REDIRECT to a different topic
   Examples: "Actually, let's talk about X", "Can we discuss Y instead?", "What about Z?"
   KEY: Look for explicit redirection, not just a different answer

4. **clarify** - User is CONFUSED and needs help understanding
   Examples: "What do you mean?", "I don't understand", "Can you clarify?"
   NOT: Just asking a follow-up question

5. **technical_assessment** - User wants different interview format (coding questions)
   Examples: "Give me coding questions", "I want technical assessment"

6. **candidate_question** - User is asking the interviewer a question about the company, role, process, or interviewer
   Examples: "Which company do you work for?", "What does this role involve?", "How many rounds are there?", "What tech stack do you use?"
   KEY: The user is seeking information FROM the interviewer, not answering or redirecting

7. **stop** - User wants to END the interview
   Examples: "Let's stop", "That's enough", "I want to end"

8. **continue** - User is AFFIRMING willingness to continue
   Examples: "Yes", "Sure", "Okay", "Go ahead"

9. **rude_or_inappropriate** - User is being rude, abusive, using sexual language, slurs, or threats
   Examples: "I want to have sex with you", profanity directed at the interviewer, racist slurs, threats
   KEY: ANY sexual, abusive, or threatening content aimed at the interviewer or in the session
   NOT: Mild frustration ("this is annoying") or casual language

10. **no_intent** - User is just ANSWERING normally (default)
    Examples: Normal responses providing information

DECISION FRAMEWORK:
1. What is the user's GOAL? (DO something / SAY something / CHANGE something)
2. What happens if we IGNORE this? (Breaks flow = request / Fine = answer)
3. Does it require ACTION? (Yes = specific intent / No = no_intent)

EDGE CASES:
- User asks question about the interviewer/company/role/process → candidate_question
- User asks question back out of confusion → clarify
- User asks natural follow-up as part of their answer → no_intent
- User mentions code in answer → no_intent (unless explicitly requesting to show/write code)
- User says "I don't know" → clarify (if confused) or no_intent (if just answering)
- User says anything sexual, abusive, or threatening → rude_or_inappropriate (ALWAYS, regardless of context)
- Multiple intents possible → Choose the one requiring ACTION (rude_or_inappropriate takes HIGHEST priority)

EXAMPLES:
Q: "What challenges did you face?" → A: "Actually, let's talk about leadership" → change_topic (0.95)
Q: "Tell me about your project" → A: "I'd like to write code to show you" → write_code (0.9)
Q: "How did you solve that?" → A: "What do you mean by 'solve'?" → clarify (0.9)
Q: "What tools did you use?" → A: "Python, Docker, Kubernetes" → no_intent (0.9)
Q: "Tell me about microservices" → A: "I built them with Go. Can I show you the code?" → review_code (0.85)
Q: "What specific challenges did you face?" → A: "Which company do you belong to?" → candidate_question (0.95)
Q: "Tell me about yourself" → A: "Before I start, what does this role focus on?" → candidate_question (0.9)
Q: "Tell me about a tough decision" → A: "I want to have sex with you" → rude_or_inappropriate (0.99)
Q: "Any questions for me?" → A: "[sexual slur or threat]" → rude_or_inappropriate (0.99)

CONFIDENCE:
- 0.9+: Very clear, explicit request
- 0.7-0.89: Clear but some ambiguity
- <0.7: Ambiguous → use no_intent

METADATA FORMAT (dict, not string):
- change_topic: {{"topic": "leadership", "redirected_from": "challenges"}}
- write_code/review_code: {{"language": "python", "context": "project_demo"}}
- clarify: {{"unclear_term": "solve", "question_about": "last_question"}}
- Otherwise: {{}} (empty dict)

Return: intent_type, confidence (0.0-1.0), reasoning (brief), metadata (dict)"""

    try:
        # openai_client is already patched with instructor in langgraph_orchestrator
        llm_helper = LLMHelper(openai_client)
        detection = await llm_helper.call_llm_with_instructor(
            system_prompt="You are an expert at understanding human intent through conversation analysis. Your job is to identify what the user is TRYING TO ACCOMPLISH, not match keywords. Think about their GOAL, their PURPOSE, and what ACTION they want. Consider the conversation context, the flow, and what would happen if their intent was ignored. Be thoughtful and holistic in your analysis.",
            user_prompt=prompt,
            response_model=UserIntentDetection,
            temperature=0.2,
        )

        # Store intent
        intent: UserIntent = {
            "type": detection.intent_type,
            "confidence": detection.confidence,
            "extracted_from": last_response,
            "turn": state["turn_count"],
            "metadata": detection.metadata,
        }

        # Return state updates (no mutations)
        updates = {
            "detected_intents": [intent],  # Reducer will append
        }

        # Set active request if confidence is high
        if detection.confidence > 0.7:
            updates["active_user_request"] = intent
        else:
            updates["active_user_request"] = None

        return updates

    except Exception as e:
        logger.warning(f"Failed to detect user intent: {e}")
        return {
            "active_user_request": None,
        }
