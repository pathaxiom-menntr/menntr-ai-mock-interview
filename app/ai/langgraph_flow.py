from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
from app.models.interview_state import InterviewState
from app.ai.llm_client import llm_client
import json
INTERVIEW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parse_resume",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_skills",
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_text": {"type": "string"}
                },
                "required": ["resume_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["question", "answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_interview_tips",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"}
                },
                "required": ["skill"]
            }
        }
    }
]


async def resume_parser(state: InterviewState) -> Dict[str, Any]:
    print("--- PARSING RESUME ---")
    return {
        "tool_calls": [{
            "id": "call_resume",
            "name": "parse_resume",
            "args": {"file_path": state.get("resume_text", "resume.pdf")}
        }],
        "next_node": "skill_extractor"
    }


async def skill_extractor(state: InterviewState) -> Dict[str, Any]:
    print("--- EXTRACTING SKILLS ---")
    return {
        "tool_calls": [{
            "id": "call_skills",
            "name": "extract_skills",
            "args": {"resume_text": state.get("resume_text", "")}
        }],
        "next_node": "generate_question"
    }


async def process_turn(state: InterviewState) -> Dict[str, Any]:
    """Evaluates last answer AND generates next question in one LLM call for zero latency."""
    print("--- EVALUATING & GENERATING ---")
    from app.services.interviewer import INTERVIEWER_SYSTEM_PROMPT
    
    # We provide context about the current question and the last answer
    messages = [
        {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT + f"\nDifficulty level: {state.get('difficulty', 'Easy')}"},
        {"role": "user", "content": (
            f"Skills: {state.get('skills')}\n"
            f"Current Question: {state.get('current_question')}\n"
            f"User's Last Answer: {state.get('last_user_input')}\n"
            f"Conversation History: {state.get('conversation_history')}\n\n"
            "Please evaluate the last answer (score 0.0-1.0 and feedback) and then provide the next question or a follow-up."
        )}
    ]

    response = await llm_client.get_completion(
        messages=messages,
        tools=INTERVIEW_TOOLS
    )

    msg = response.choices[0].message
    
    # Process potential tool calls (like tips)
    if msg.tool_calls:
        return {
            "tool_calls": [{
                "id": msg.tool_calls[0].id,
                "name": msg.tool_calls[0].function.name,
                "args": json.loads(msg.tool_calls[0].function.arguments)
            }],
            "next_node": "process_turn"
        }

    # Extract evaluation and question from the message content
    full_content = msg.content or ""
    display_content = full_content
    eval_data = {"score": 0.8, "feedback": "Constructive feedback pending."}
    
    if "---" in full_content:
        parts = full_content.split("---")
        display_content = parts[0].strip()
        json_part = parts[-1].strip()
        try:
            # Clean up potential markdown code blocks around JSON
            if "```json" in json_part:
                json_part = json_part.split("```json")[-1].split("```")[0].strip()
            elif "```" in json_part:
                json_part = json_part.split("```")[-1].split("```")[0].strip()
            
            import json as json_lib
            extracted = json_lib.loads(json_part)
            if isinstance(extracted, dict):
                eval_data["score"] = float(extracted.get("score", 0.8))
                eval_data["feedback"] = str(extracted.get("feedback", "Good answer."))
        except Exception as e:
            print(f"Error parsing evaluation JSON: {e}")

    # Update score if we have an answer to evaluate
    from app.services.scoring import scoring_service
    new_answers = state.get("answers", [])
    if state.get("last_user_input") and state.get("current_question"):
        new_answers.append({
            "question_id": state["current_question"].get("id", "1"),
            "text": state["last_user_input"],
            "score": eval_data["score"],
            "feedback": eval_data["feedback"]
        })

    next_q = {
        "id": str(len(state.get("questions", [])) + 1),
        "text": display_content,
        "skill": "General",
        "difficulty": state.get("difficulty", "Easy")
    }

    return {
        "current_question": next_q,
        "questions": state.get("questions", []) + [next_q],
        "answers": new_answers,
        "total_score": scoring_service.calculate_overall_score(new_answers),
        "tool_calls": []
    }


def get_user_answer(state: InterviewState) -> Dict[str, Any]:
    print("--- GET USER ANSWER ---")
    return {"last_user_input": state.get("last_user_input", "")}


async def evaluate_answer_node(state: InterviewState) -> Dict[str, Any]:
    print("--- EVALUATING ANSWER ---")

    return {
        "tool_calls": [{
            "id": "call_eval",
            "name": "evaluate_answer",
            "args": {
                "question": state["current_question"]["text"],
                "answer": state["last_user_input"]
            }
        }],
        "next_node": "update_score"
    }


def update_score(state: InterviewState) -> Dict[str, Any]:
    print("--- UPDATE SCORE ---")
    from app.services.scoring import scoring_service
    return {
        "total_score": scoring_service.calculate_overall_score(state.get("answers", []))
    }


def decide_next(state: InterviewState):
    print("--- DECIDING NEXT STEP ---")

    answers = state.get("answers", [])

    if not isinstance(answers, list):
        return END

    print("Answers count:", len(answers))

    if len(answers) >= 5:
        return END

    return "generate_question"

async def tool_node(state: InterviewState) -> Dict[str, Any]:
    print("--- TOOL EXECUTION ---")

    from app.ai.mcp_server import (
        parse_resume,
        extract_skills,
        evaluate_answer,
        get_interview_tips
    )

    updates: Dict[str, Any] = {
        "tool_outputs": [],
        "tool_calls": []  
    }

    for call in state.get("tool_calls", []):

        if call["name"] == "parse_resume":
            result = await parse_resume(call["args"]["file_path"])
            updates["resume_text"] = result

        elif call["name"] == "extract_skills":
            result = await extract_skills(call["args"]["resume_text"])
            try:
                updates["skills"] = json.loads(result)
            except:
                updates["skills"] = ["General"]

        elif call["name"] == "evaluate_answer":
            result = await evaluate_answer(
                call["args"]["question"],
                call["args"]["answer"]
            )

            try:
                eval_data = json.loads(result)
            except:
                eval_data = {"score": 0.5, "feedback": result}

            updates["answers"] = state.get("answers", []) + [{
                "question_id": state["current_question"]["id"],
                "text": state["last_user_input"],
                "score": eval_data["score"],
                "feedback": eval_data["feedback"]
            }]

        elif call["name"] == "get_interview_tips":
            result = await get_interview_tips(call["args"]["skill"])
            updates["tool_outputs"].append({
                "call_id": call["id"],
                "content": result
            })

    return updates

workflow = StateGraph(InterviewState)

def route_tool_or_next(next_node):
    def route(state):
        return "tools" if state.get("tool_calls") else next_node
    return route

workflow.add_node("resume_parser", resume_parser)
workflow.add_node("skill_extractor", skill_extractor)
workflow.add_node("process_turn", process_turn)
workflow.add_node("tools", tool_node)

# Conditional Entry Point
def entry_point(state):
    if state.get("last_user_input"):
        return "process_turn"
    return "resume_parser"

workflow.add_conditional_edges(START, entry_point)

workflow.add_conditional_edges(
    "resume_parser",
    route_tool_or_next("skill_extractor"),
    {"tools": "tools", "skill_extractor": "skill_extractor"}
)

workflow.add_conditional_edges(
    "skill_extractor",
    route_tool_or_next("process_turn"),
    {"tools": "tools", "process_turn": "process_turn"}
)

workflow.add_conditional_edges(
    "process_turn",
    route_tool_or_next(END),
    {"tools": "tools", END: END}
)

workflow.add_conditional_edges(
    "tools",
    lambda state: state.get("next_node", "process_turn"),
    {
        "skill_extractor": "skill_extractor",
        "process_turn": "process_turn"
    }
)
app_graph = workflow.compile()
# Config for app_graph can be set during invocation or here optionally
# config = {"recursion_limit": 50}
