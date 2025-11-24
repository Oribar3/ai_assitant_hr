"""Pydantic models for job descriptions."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RequiredSkills(BaseModel):
    """Required skills for a job position."""
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


class PreferredSkills(BaseModel):
    """Preferred (nice-to-have) skills for a job position."""
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    """Job description with requirements."""
    job_title: str
    company: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    job_type: Optional[str] = None  # Full-time, Part-time, Contract
    
    summary: str
    responsibilities: List[str] = Field(default_factory=list)
    
    required_skills: RequiredSkills
    preferred_skills: Optional[PreferredSkills] = None
    
    min_years_experience: Optional[int] = None
    max_years_experience: Optional[int] = None
    
    education_requirements: List[str] = Field(default_factory=list)
    
    salary_range: Optional[str] = None
    benefits: List[str] = Field(default_factory=list)
    
    company_culture: Optional[str] = None
    raw_text: Optional[str] = None  # Original job description
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_title": "Senior Python Developer",
                "company": "Tech Corp",
                "summary": "We are seeking an experienced Python developer...",
                "required_skills": {
                    "technical_skills": ["Python", "Django", "PostgreSQL", "REST APIs"],
                    "soft_skills": ["Communication", "Team player"],
                    "languages": ["English"]
                },
                "min_years_experience": 5,
                "education_requirements": ["Bachelor's in Computer Science or related field"]
            }
        }
