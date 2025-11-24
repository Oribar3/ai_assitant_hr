"""Pydantic models for evaluation results."""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown for a candidate."""
    skills_match: float = Field(..., ge=0, le=100)
    experience_relevance: float = Field(..., ge=0, le=100)
    education_match: float = Field(..., ge=0, le=100)
    cultural_fit: float = Field(..., ge=0, le=100)
    overall_score: float = Field(..., ge=0, le=100)


class MatchAnalysis(BaseModel):
    """Analysis of how well candidate matches job requirements."""
    matched_required_skills: List[str] = Field(default_factory=list)
    missing_required_skills: List[str] = Field(default_factory=list)
    matched_preferred_skills: List[str] = Field(default_factory=list)
    additional_relevant_skills: List[str] = Field(default_factory=list)
    
    experience_match: str  # "Exceeds", "Meets", "Below" requirements
    education_match: str  # "Exceeds", "Meets", "Below" requirements
    
    similarity_score: float = Field(..., ge=0, le=1)  # Semantic similarity


class EvaluationResult(BaseModel):
    """Complete evaluation result for a candidate."""
    candidate_name: str
    job_title: str
    
    scores: ScoreBreakdown
    match_analysis: MatchAnalysis
    
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    
    recommendation: str  # "STRONG HIRE", "HIRE", "MAYBE", "PASS"
    reasoning: str
    
    suggested_interview_questions: List[str] = Field(default_factory=list)
    
    evaluation_date: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "candidate_name": "John Doe",
                "job_title": "Senior Python Developer",
                "scores": {
                    "skills_match": 95.0,
                    "experience_relevance": 90.0,
                    "education_match": 85.0,
                    "cultural_fit": 88.0,
                    "overall_score": 92.0
                },
                "recommendation": "STRONG HIRE",
                "reasoning": "Excellent technical match with strong leadership experience"
            }
        }


class CandidateReport(BaseModel):
    """Summary report for multiple candidates."""
    job_title: str
    company: Optional[str] = None
    total_candidates: int
    evaluation_date: datetime = Field(default_factory=datetime.now)
    
    top_candidates: List[EvaluationResult] = Field(default_factory=list)
    
    summary: Optional[str] = None
    hiring_recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_title": "Senior Python Developer",
                "total_candidates": 10,
                "top_candidates": [],
                "summary": "Analysis of 10 candidates for Senior Python Developer position"
            }
        }
