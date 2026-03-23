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


async def generate_question(state: InterviewState) -> Dict[str, Any]:
    print("--- GENERATING QUESTION ---")

    messages = [
        {"role": "system", "content": "You are an expert technical interviewer."},
        {"role": "user", "content": f"Skills: {state.get('skills')}\nPrevious Questions: {state.get('questions')}"}
    ]

    response = await llm_client.get_completion(
        messages=messages,
        tools=INTERVIEW_TOOLS
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        return {
            "tool_calls": [{
                "id": msg.tool_calls[0].id,
                "name": msg.tool_calls[0].function.name,
                "args": json.loads(msg.tool_calls[0].function.arguments)
            }],
            "next_node": "generate_question"
        }

    question = {
        "id": str(len(state.get("questions", [])) + 1),
        "text": msg.content,
        "skill": "General",
        "difficulty": "Medium"
    }

    return {
        "current_question": question,
        "questions": state.get("questions", []) + [question],
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

workflow.add_node("resume_parser", resume_parser)
workflow.add_node("skill_extractor", skill_extractor)
workflow.add_node("generate_question", generate_question)
workflow.add_node("get_user_answer", get_user_answer)
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("update_score", update_score)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "resume_parser")


def route_tool_or_next(next_node):
    def route(state):
        return "tools" if state.get("tool_calls") else next_node
    return route


workflow.add_conditional_edges(
    "resume_parser",
    route_tool_or_next("skill_extractor"),
    {"tools": "tools", "skill_extractor": "skill_extractor"}
)

workflow.add_conditional_edges(
    "skill_extractor",
    route_tool_or_next("generate_question"),
    {"tools": "tools", "generate_question": "generate_question"}
)

workflow.add_conditional_edges(
    "generate_question",
    route_tool_or_next("get_user_answer"),
    {"tools": "tools", "get_user_answer": "get_user_answer"}
)

workflow.add_conditional_edges(
    "evaluate_answer",
    route_tool_or_next("update_score"),
    {"tools": "tools", "update_score": "update_score"}
)

workflow.add_edge("get_user_answer", "evaluate_answer")

workflow.add_conditional_edges(
    "update_score",
    decide_next,
    {
        "generate_question": "generate_question",
        END: END
    }
)

workflow.add_conditional_edges(
    "tools",
    lambda state: state.get("next_node", "generate_question"),
    {
        "skill_extractor": "skill_extractor",
        "generate_question": "generate_question",
        "update_score": "update_score"
    }
)
app_graph = workflow.compile()
# Config for app_graph can be set during invocation or here optionally
# config = {"recursion_limit": 50}