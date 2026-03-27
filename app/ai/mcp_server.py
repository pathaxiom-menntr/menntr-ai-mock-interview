import asyncio
from mcp.server.fastmcp import FastMCP
from app.services.resume_parser import resume_parser_service
from app.ai.llm_client import llm_client
import json

mcp = FastMCP("Interview Assistant")
@mcp.tool()
async def parse_resume(file_path: str) -> str:
    """
    Parse a resume file (PDF, DOCX, or Text) and extract its content.
    
    Args:
        file_path: Absolute path to the resume file.
    """
    try:
        content = resume_parser_service.parse(file_path)
        return content
    except Exception as e:
        return f"Error parsing resume: {str(e)}"
@mcp.tool()
async def extract_skills(resume_text: str) -> str:
    """
    Extract a list of top 5 technical skills from a given resume text using LLM analysis.
    
    Args:
        resume_text: The full text content parsed from a resume.
    """
    messages = [
        {"role": "system", "content": "You are a recruitment expert. Extract the top 5 technical skills from the resume text provided. Return a JSON list of strings: [\"Skill1\", \"Skill2\", ...]"},
        {"role": "user", "content": f"Resume Text: {resume_text}"}
    ]
    
    response = await llm_client.get_completion(messages=messages)
    skills_json = response.choices[0].message.content
    return skills_json




@mcp.tool()
async def evaluate_answer(question: str, answer: str) -> str:
    """
    Evaluate a user's answer to a technical question using LLM analysis.
    
    Args:
        question: The interview question asked.
        answer: The user's response to the question.
    """
    messages = [
        {"role": "system", "content": "You are an expert interviewer. Evaluate the user's answer to the technical question. Provide a score (0.0 to 1.0) and short feedback in JSON format: {\"score\": 0.8, \"feedback\": \"...\"}"},
        {"role": "user", "content": f"Question: {question}\nUser Answer: {answer}"}
    ]
    response = await llm_client.get_completion(messages=messages)
    eval_content = response.choices[0].message.content
    return eval_content

@mcp.tool()
async def get_interview_tips(skill: str) -> str:
    """
    Get specific interview tips and common questions for a given technical skill.
    
    Args:
        skill: The technical skill (e.g., 'Python', 'FastAPI', 'React')
    """
    tips = {
        "python": "Focus on decorators, generators, and memory management. Be ready to explain GIL.",
        "fastapi": "Be prepared to discuss Pydantic models, dependency injection, and async/await benefits.",
        "react": "Understand hooks (useEffect/useMemo), virtual DOM, and state management patterns.",
        "ai": "Explain the difference between supervised and unsupervised learning. Know about transformers."
    }
    
    skill_lower = skill.lower()
    if skill_lower in tips:
        return f"Tips for {skill}: {tips[skill_lower]}"
    else:
        return f"No specific tips found for {skill}, but general advice is to focus on core fundamentals and project experience."

if __name__ == "__main__":
    mcp.run()
