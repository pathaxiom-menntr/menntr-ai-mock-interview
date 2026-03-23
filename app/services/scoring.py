from typing import List
from app.models.interview_state import Answer

class ScoringService:
    def calculate_overall_score(self, answers: List[Answer]) -> float:
        """
        Calculate simple average of all answer scores.
        """
        if not answers:
            return 0.0
        
        # In InterviewState, answers is List[Answer] and each Answer is a Pydantic model
        # but in LangGraph it might be passed as a list of dicts or objects.
        # Handle both cases.
        total = 0.0
        count = 0
        
        for ans in answers:
            if isinstance(ans, dict):
                score = ans.get("score", 0.0)
            else:
                score = getattr(ans, "score", 0.0)
            
            total += score
            count += 1
            
        return total / count if count > 0 else 0.0

scoring_service = ScoringService()
