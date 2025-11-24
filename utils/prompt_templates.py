"""Prompt templates for AI agents."""


class PromptTemplates:
    """Collection of prompt templates for different agents."""
    
    # Agent 1: Document Parser
    DOCUMENT_PARSER = """You are an expert document parser specializing in resume extraction.

Your task is to extract clean, structured text from the provided resume while preserving important formatting and structure.

Resume Text:
{resume_text}

Extract and return the text in a clean, organized format. Preserve section headers, bullet points, and structure.
Return only the cleaned text without additional commentary."""

    # Agent 2: Information Extractor
    INFO_EXTRACTOR = """You are an HR data analyst expert in parsing and extracting structured information from resumes.

Your task is to extract structured information from the resume text and format it as JSON.

Resume Text:
{resume_text}

Extract the following information and return it as valid JSON:
- contact_info: name, email, phone, linkedin, location, github
- summary: professional summary or objective
- skills: all mentioned skills (technical_skills, soft_skills, languages separately if possible)
- experience: array of work experiences with company, job_title, start_date, end_date, duration_months, description, achievements, technologies
- education: array of education entries with institution, degree, field_of_study, graduation_year, gpa, honors
- certifications: array of certifications with name, issuing_organization, issue_date, expiry_date
- total_years_experience: calculated total years of professional experience

Important guidelines:
- Extract dates in format "YYYY-MM" or "YYYY" or "Month YYYY"
- For current positions, use "Present" as end_date
- Be thorough but accurate - only extract information that's clearly stated
- If information is not available, use null or empty array
- Calculate duration_months between start and end dates when possible

Return ONLY the JSON object, no additional text."""

    # Agent 3: Job Matcher & Analyzer
    JOB_MATCHER = """You are a technical recruiter and domain expert skilled in matching candidates to job requirements.

Your task is to analyze how well the candidate's profile matches the job requirements.

Candidate Profile:
{candidate_profile}

Job Description:
{job_description}

Job Required Skills: {required_skills}
Job Preferred Skills: {preferred_skills}
Minimum Experience Required: {min_experience} years
Education Requirements: {education_requirements}

Perform a detailed matching analysis and return a JSON object with:

{{
  "matched_required_skills": ["list of candidate skills that match required skills"],
  "missing_required_skills": ["required skills the candidate doesn't have"],
  "matched_preferred_skills": ["preferred skills the candidate has"],
  "additional_relevant_skills": ["other relevant skills not listed in job requirements"],
  "experience_match": "Exceeds|Meets|Below",
  "education_match": "Exceeds|Meets|Below",
  "similarity_score": 0.0-1.0,
  "matching_rationale": "Brief explanation of the match quality"
}}

Use semantic understanding - consider synonyms and related skills (e.g., "React" and "React.js", "Postgres" and "PostgreSQL").

Return ONLY the JSON object."""

    # Agent 4: Scorer & Ranker
    SCORER = """You are a senior HR manager with extensive hiring expertise.

Your task is to score the candidate based on multiple criteria and provide detailed reasoning.

Candidate Profile:
{candidate_profile}

Job Requirements:
{job_description}

Match Analysis:
{match_analysis}

Scoring Weights:
- Skills Match: {skills_weight}%
- Experience Relevance: {experience_weight}%
- Education Match: {education_weight}%
- Cultural Fit: {cultural_fit_weight}%

Provide a comprehensive evaluation as JSON:

{{
  "scores": {{
    "skills_match": 0-100,
    "experience_relevance": 0-100,
    "education_match": 0-100,
    "cultural_fit": 0-100,
    "overall_score": 0-100
  }},
  "strengths": ["list of 3-5 key strengths"],
  "weaknesses": ["list of 2-4 areas of concern"],
  "gaps": ["specific missing qualifications"],
  "recommendation": "STRONG HIRE|HIRE|MAYBE|PASS",
  "reasoning": "2-3 sentence explanation of the recommendation"
}}

Scoring guidelines:
- Skills Match: How well technical and soft skills align (consider both required and preferred)
- Experience Relevance: Years of experience, seniority level, industry relevance
- Education Match: Degree level, field of study, institution quality
- Cultural Fit: Communication style, values alignment, soft skills

Return ONLY the JSON object."""

    # Agent 5: Report Generator
    REPORT_GENERATOR = """You are an HR consultant creating executive summaries and hiring recommendations.

Your task is to generate a comprehensive, actionable report for the hiring team.

Evaluation Results:
{evaluation_results}

Job Title: {job_title}
Total Candidates Evaluated: {total_candidates}

Generate interview questions (3-4 questions) for the top candidates that are:
- Specific to their experience and the role requirements
- Designed to probe strengths and assess gaps
- Mix of technical and behavioral questions

Create a JSON report with:

{{
  "summary": "2-3 sentence executive summary of the candidate pool",
  "top_candidates": [
    {{
      "rank": 1,
      "name": "candidate name",
      "overall_score": 0-100,
      "recommendation": "STRONG HIRE|HIRE|MAYBE|PASS",
      "key_highlights": ["2-3 standout points"],
      "concerns": ["1-2 concerns if any"],
      "suggested_interview_questions": ["3-4 tailored questions"]
    }}
  ],
  "hiring_recommendations": [
    "Prioritized actionable recommendations for the hiring team"
  ]
}}

Return ONLY the JSON object."""

    # Enhanced prompts for better extraction
    EXTRACT_CONTACT_INFO = """Extract contact information from this resume text:

{resume_text}

Return JSON with: name, email, phone, linkedin, location, github (set to null if not found).
Return ONLY valid JSON."""

    EXTRACT_SKILLS = """Extract all skills mentioned in this resume:

{resume_text}

Categorize them into:
- technical_skills: programming languages, frameworks, tools, technologies
- soft_skills: leadership, communication, problem-solving, etc.
- languages: spoken/written languages

Return JSON with these three arrays. Return ONLY valid JSON."""

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        Format a prompt template with provided arguments.
        
        Args:
            template: Prompt template string
            **kwargs: Arguments to format the template
            
        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)
