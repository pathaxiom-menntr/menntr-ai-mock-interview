from typing import List, Optional, TypedDict, Dict
from pydantic import BaseModel


class Question(BaseModel):
    id: str
    text: str
    skill: str
    difficulty: str


class Answer(BaseModel):
    question_id: str
    text: str
    score: float
    feedback: str


class InterviewState(TypedDict):
    # User info
    user_id: Optional[str]
    resume_text: Optional[str]
    skills: Optional[List[str]]
    
    # Interview progress
    questions: Optional[List[Question]]
    current_question: Optional[Question]
    answers: Optional[List[Answer]]
    
    # Tool state
    tool_calls: Optional[List[Dict]]
    tool_outputs: Optional[List[Dict]]
    
    # Session state
    total_score: Optional[float]
    is_finished: Optional[bool]
    last_user_input: Optional[str]
    next_node: Optional[str]