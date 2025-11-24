"""Pydantic models for candidate data."""

from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import date


class ContactInfo(BaseModel):
    """Contact information for a candidate."""
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    location: Optional[str] = None
    github: Optional[str] = None


class Experience(BaseModel):
    """Work experience entry."""
    company: str
    job_title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None  # "Present" for current jobs
    duration_months: Optional[int] = None
    description: Optional[str] = None
    achievements: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)


class Education(BaseModel):
    """Education entry."""
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    graduation_year: Optional[int] = None
    gpa: Optional[str] = None
    honors: Optional[str] = None


class Certification(BaseModel):
    """Professional certification."""
    name: str
    issuing_organization: str
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None


class Candidate(BaseModel):
    """Complete candidate profile extracted from resume."""
    contact_info: ContactInfo
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    total_years_experience: Optional[float] = None
    raw_text: Optional[str] = None  # Original resume text
    
    class Config:
        json_schema_extra = {
            "example": {
                "contact_info": {
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1-555-0123",
                    "linkedin": "linkedin.com/in/johndoe",
                    "location": "San Francisco, CA"
                },
                "summary": "Senior software engineer with 8 years of experience...",
                "technical_skills": ["Python", "Django", "FastAPI", "PostgreSQL"],
                "soft_skills": ["Leadership", "Communication", "Problem Solving"],
                "languages": ["English", "Spanish"],
                "total_years_experience": 8.0
            }
        }
