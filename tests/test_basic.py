"""Basic tests for the resume screening pipeline."""

import os
import pytest
from pathlib import Path

from utils.file_handler import FileHandler
from models.candidate import Candidate, ContactInfo
from models.job_description import JobDescription
from models.evaluation import EvaluationResult, ScoreBreakdown


def test_file_handler_validation():
    """Test file handler validation."""
    handler = FileHandler()

    # Test with existing file
    test_file = Path("tests/sample_data/sample_resume.txt")
    is_valid, error_msg = handler.validate_file(str(test_file))
    assert is_valid == True
    assert error_msg == ""

    # Test with non-existent file
    is_valid, error_msg = handler.validate_file("non_existent_file.pdf")
    assert is_valid == False
    assert "File not found" in error_msg

    # Test with unsupported format
    is_valid, error_msg = handler.validate_file("tests/sample_data/sample_job_description.json")
    assert is_valid == False
    assert "Unsupported file format" in error_msg


def test_file_handler_read_resume():
    """Test reading a resume file."""
    handler = FileHandler()

    test_file = "tests/sample_data/sample_resume.txt"
    content = handler.read_resume(test_file)

    assert isinstance(content, str)
    assert len(content) > 100  # Should have substantial content
    assert "John Smith" in content  # Should contain the name
    assert "Python" in content  # Should contain skills


def test_candidate_model_creation():
    """Test creating a Candidate model."""
    contact_info = ContactInfo(
        name="John Doe",
        email="john.doe@example.com",
        phone="+1234567890",
        location="San Francisco, CA"
    )

    candidate_data = {
        "contact_info": contact_info,
        "summary": "Experienced developer",
        "skills": ["Python", "Django", "React"],
        "technical_skills": ["Python", "Django", "PostgreSQL"],
        "soft_skills": ["Communication", "Leadership"],
        "languages": ["English", "Spanish"],
        "experience": [],
        "education": [],
        "certifications": [],
        "total_years_experience": 5.0
    }

    candidate = Candidate(**candidate_data)

    assert candidate.contact_info.name == "John Doe"
    assert "Python" in candidate.technical_skills
    assert candidate.total_years_experience == 5.0


def test_job_description_model_creation():
    """Test creating a JobDescription model."""
    required_skills = {
        "technical_skills": ["Python", "Django"],
        "soft_skills": ["Communication"],
        "certifications": [],
        "languages": ["English"]
    }

    job_data = {
        "job_title": "Senior Python Developer",
        "company": "Tech Corp",
        "summary": "Looking for experienced Python developer",
        "responsibilities": ["Develop web applications", "Mentor juniors"],
        "required_skills": required_skills,
        "min_years_experience": 5,
        "education_requirements": ["Bachelor's degree"],
        "company_culture": "Innovative and collaborative"
    }

    job = JobDescription(**job_data)

    assert job.job_title == "Senior Python Developer"
    assert "Python" in job.required_skills.technical_skills
    assert job.min_years_experience == 5


def test_evaluation_result_creation():
    """Test creating an EvaluationResult model."""
    scores = ScoreBreakdown(
        skills_match=95.0,
        experience_relevance=90.0,
        education_match=85.0,
        cultural_fit=88.0,
        overall_score=92.0
    )

    evaluation_data = {
        "candidate_name": "John Doe",
        "job_title": "Senior Python Developer",
        "scores": scores,
        "match_analysis": {},
        "strengths": ["Strong Python skills", "Leadership experience"],
        "weaknesses": ["Limited cloud experience"],
        "gaps": ["AWS knowledge"],
        "recommendation": "STRONG HIRE",
        "reasoning": "Excellent technical fit with demonstrated experience"
    }

    evaluation = EvaluationResult(**evaluation_data)

    assert evaluation.candidate_name == "John Doe"
    assert evaluation.scores.overall_score == 92.0
    assert evaluation.recommendation == "STRONG HIRE"
    assert len(evaluation.strengths) == 2


def test_file_handler_output():
    """Test saving reports."""
    handler = FileHandler()

    test_content = {"test": "data", "score": 95}
    test_path = "tests/sample_data/test_output.json"

    # Ensure cleanup even if test fails
    try:
        import json
        handler.save_report(json.dumps(test_content, indent=2), test_path)

        # Verify file was created
        assert Path(test_path).exists()

        # Verify content
        with open(test_path, 'r') as f:
            loaded_data = json.loads(f.read())
            assert loaded_data["test"] == "data"
            assert loaded_data["score"] == 95

    finally:
        # Clean up
        if Path(test_path).exists():
            os.remove(test_path)


def test_config_loading():
    """Test loading configuration."""
    from core.chain_orchestrator import ChainConfig

    config = ChainConfig.from_yaml("config.yaml")

    # Should have loaded the default sections
    assert hasattr(config, 'llm_config')
    assert hasattr(config, 'scoring_config')
    assert 'weights' in config.scoring_config

    # Weights should sum to 1.0
    weights = config.scoring_config["weights"]
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.001


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
