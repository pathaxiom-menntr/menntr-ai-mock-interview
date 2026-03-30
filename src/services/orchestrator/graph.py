"""LangGraph StateGraph definition for interview orchestration.

Graph structure (new):

    START
      ↓
    ingest_input
      ↓ (conditional)
    ┌─────────────────────────────────┐
    │  plan_interview  ← FIRST TURN   │  (generates blueprint from resume + JD)
    │  greeting        ← FIRST TURN   │
    │  code_review     ← code submit  │
    │  detect_intent   ← normal turn  │
    └─────────────────────────────────┘
      ↓
    detect_intent → decide_next_action
      ↓ (conditional, driven by depth engine)
    [action nodes: greeting, answer_candidate_question, question, followup,
                   sandbox_guidance, code_review, evaluation, closing]
      ↓
    finalize_turn  (single writer for conversation_history; updates plan topic state)
      ↓
    END
"""

import logging
from typing import Literal, Tuple
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.services.orchestrator.types import InterviewState
from src.services.orchestrator.nodes import NodeHandler

logger = logging.getLogger(__name__)


def create_interview_graph(node_handler: NodeHandler) -> Tuple[StateGraph, MemorySaver]:
    """Create and compile the LangGraph StateGraph."""
    graph = StateGraph(InterviewState)

    # ── Register all nodes ──────────────────────────────────────────────────
    graph.add_node("ingest_input",               node_handler.ingest_input_node)
    graph.add_node("plan_interview",             node_handler.plan_interview_node)
    graph.add_node("detect_intent",              node_handler.detect_intent_node)
    graph.add_node("decide_next_action",         node_handler.decide_next_action_node)
    graph.add_node("greeting",                   node_handler.greeting_node)
    graph.add_node("answer_candidate_question",  node_handler.answer_candidate_question_node)
    graph.add_node("question",                   node_handler.question_node)
    graph.add_node("followup",                   node_handler.followup_node)
    graph.add_node("sandbox_guidance",           node_handler.sandbox_guidance_node)
    graph.add_node("code_review",                node_handler.code_review_node)
    graph.add_node("evaluation",                 node_handler.evaluation_node)
    graph.add_node("closing",                    node_handler.closing_node)
    graph.add_node("termination",                node_handler.termination_node)
    graph.add_node("finalize_turn",              node_handler.finalize_turn_node)

    graph.set_entry_point("ingest_input")

    # ── Routing from ingest_input ────────────────────────────────────────────
    def route_from_ingest(
        state: InterviewState,
    ) -> Literal["plan_interview", "code_review", "detect_intent"]:
        """
        First turn: run plan_interview (generates blueprint) then greeting.
        Code submitted: jump straight to code_review.
        Normal turn: detect_intent → decide_next_action.
        """
        conv_history = state.get("conversation_history", [])
        has_greeting = any(
            msg.get("role") == "assistant" and msg.get("content")
            for msg in conv_history
        )

        # First-ever turn: need to plan and greet
        if not has_greeting and state.get("turn_count", 0) == 0:
            return "plan_interview"

        # Code was submitted
        if state.get("current_code"):
            return "code_review"

        return "detect_intent"

    graph.add_conditional_edges(
        "ingest_input",
        route_from_ingest,
        {
            "plan_interview": "plan_interview",
            "code_review": "code_review",
            "detect_intent": "detect_intent",
        },
    )

    # plan_interview always flows to greeting on first turn
    graph.add_edge("plan_interview", "greeting")

    # Normal conversation flow
    graph.add_edge("detect_intent", "decide_next_action")

    # ── Routing from decide_next_action ─────────────────────────────────────
    graph.add_conditional_edges(
        "decide_next_action",
        route_action_node,
        {
            "greeting":                  "greeting",
            "answer_candidate_question": "answer_candidate_question",
            "question":                  "question",
            "followup":                  "followup",
            "sandbox_guidance":          "sandbox_guidance",
            "code_review":               "code_review",
            "evaluation":                "evaluation",
            "closing":                   "closing",
            "termination":               "termination",
        },
    )

    # ── All action nodes → finalize_turn → END ───────────────────────────────
    for node in [
        "greeting", "answer_candidate_question", "question", "followup",
        "sandbox_guidance", "code_review", "evaluation", "closing", "termination",
    ]:
        graph.add_edge(node, "finalize_turn")

    graph.add_edge("finalize_turn", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled, checkpointer


def route_action_node(state: InterviewState) -> Literal[
    "greeting", "answer_candidate_question", "question", "followup",
    "sandbox_guidance", "code_review", "evaluation", "closing", "termination"
]:
    """Route to action node based on next_node in state."""
    action = state.get("next_node")

    valid_actions = {
        "greeting", "answer_candidate_question", "question", "followup",
        "sandbox_guidance", "code_review", "evaluation", "closing", "termination",
    }

    if not action:
        logger.error(
            f"next_node is None. last_node={state.get('last_node')}, "
            f"state keys={sorted(state.keys())}"
        )
        return "question"

    if action not in valid_actions:
        logger.error(f"Invalid action '{action}', defaulting to 'question'.")
        return "question"

    return action
