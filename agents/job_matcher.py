"""Agent 3: Job Matcher - Analyzes candidate fit against job requirements."""

import json
from typing import Dict, Any, List, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from models.candidate import Candidate
from models.job_description import JobDescription
from utils.prompt_templates import PromptTemplates
from utils.vector_store import VectorStoreManager


class JobMatcherAgent:
    """
    Third agent in the chain: Matches candidates to job requirements.
    
    Role: Technical recruiter and domain expert
    Task: Analyze candidate profile against job requirements using semantic matching
    """

    def __init__(self, llm_config: dict = None, vector_config: dict = None):
        """
        Initialize the job matcher agent.

        Args:
            llm_config: Configuration for the language model
            vector_config: Configuration for vector store
        """
        self.llm_config = llm_config or {}
        self.vector_config = vector_config or {}
        self.llm = self._initialize_llm()
        self.vector_store = self._initialize_vector_store()
        self.prompt_template = PromptTemplates.JOB_MATCHER
        logger.info("Job Matcher Agent initialized")

    def _initialize_llm(self):
        """Initialize the language model."""
        default_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,  # Slightly higher temperature for analysis
            "max_tokens": 3000,
        }
        config = {**default_config, **self.llm_config}
        return ChatOpenAI(**config)

    def _initialize_vector_store(self):
        """Initialize the vector store for semantic matching."""
        model_name = self.vector_config.get("model_name", "all-MiniLM-L6-v2")
        return VectorStoreManager(model_name)

    def analyze_job_match(
        self,
        candidate: Candidate,
        job_description: JobDescription
    ) -> Dict[str, Any]:
        """
        Analyze how well candidate matches job requirements.

        Args:
            candidate: Candidate profile from Agent 2
            job_description: Job requirements

        Returns:
            Dictionary containing matching analysis
        """
        logger.info("Starting job matching analysis")

        try:
            # Extract data for semantic matching
            candidate_skills = candidate.technical_skills + candidate.soft_skills + candidate.languages
            required_skills = job_description.required_skills.technical_skills + \
                            job_description.required_skills.soft_skills + \
                            job_description.required_skills.languages
            preferred_skills = []
            if job_description.preferred_skills:
                preferred_skills = job_description.preferred_skills.technical_skills + \
                                 job_description.preferred_skills.soft_skills + \
                                 job_description.preferred_skills.languages

            # Perform semantic skill matching
            required_matches, required_details = self.vector_store.find_matches(
                required_skills, candidate_skills, threshold=0.6
            )

            preferred_matches, preferred_details = self.vector_store.find_matches(
                preferred_skills, candidate_skills, threshold=0.6
            ) if preferred_skills else ([], [])

            # Calculate similarity score using LLM
            similarity_analysis = self._perform_similarity_analysis(
                candidate, job_description, required_matches, preferred_matches
            )

            # Analyze experience match
            experience_match = self._analyze_experience_match(
                candidate, job_description
            )

            # Analyze education match
            education_match = self._analyze_education_match(
                candidate, job_description
            )

            # Compile final analysis
            analysis = {
                "matched_required_skills": required_matches,
                "missing_required_skills": [
                    skill for skill in required_skills
                    if skill not in required_matches
                ],
                "matched_preferred_skills": preferred_matches,
                "additional_relevant_skills": [
                    skill for skill in candidate_skills
                    if skill not in (required_matches + preferred_matches)
                    and skill != skill  # Placeholder logic, will be refined
                ],
                "experience_match": experience_match,
                "education_match": education_match,
                "similarity_score": similarity_analysis.get("similarity_score", 0.0),
                "matching_rationale": similarity_analysis.get("rationale", "")
            }

            logger.info("Job matching analysis completed successfully")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze job match: {e}")
            return self._create_fallback_analysis()

    def _perform_similarity_analysis(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        required_matches: List[str],
        preferred_matches: List[str]
    ) -> Dict[str, Any]:
        """
        Use LLM to perform deep similarity analysis.

        Args:
            candidate: Candidate profile
            job_description: Job description
            required_matches: Matched required skills
            preferred_matches: Matched preferred skills

        Returns:
            Similarity analysis results
        """
        try:
            # Format the prompt
            prompt = PromptTemplates.format_prompt(
                self.prompt_template,
                candidate_profile=json.dumps({
                    "summary": candidate.summary,
                    "technical_skills": candidate.technical_skills,
                    "soft_skills": candidate.soft_skills,
                    "languages": candidate.languages,
                    "total_years_experience": candidate.total_years_experience,
                    "experience": [{"job_title": exp.job_title, "company": exp.company}
                                 for exp in candidate.experience[:3]] if candidate.experience else []
                }, indent=2),
                job_description=str({
                    "title": job_description.job_title,
                    "summary": job_description.summary,
                    "responsibilities": job_description.responsibilities[:3] if job_description.responsibilities else []
                }),
                required_skills=required_matches,
                preferred_skills=preferred_matches,
                min_experience=job_description.min_years_experience or 0,
                education_requirements=job_description.education_requirements or []
            )

            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template(prompt)

            # Run analysis
            chain = prompt_template | self.llm
            response = chain.invoke({})

            # Parse JSON response
            response_text = response.content.strip()
            analysis = self._parse_analysis_response(response_text)

            return analysis

        except Exception as e:
            logger.error(f"Failed to perform similarity analysis: {e}")
            return {
                "similarity_score": 0.5,
                "rationale": "Analysis failed, using default similarity score"
            }

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM analysis."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass

        # Fallback
        return {
            "similarity_score": 0.5,
            "matching_rationale": response_text[:200]
        }

    def _analyze_experience_match(
        self,
        candidate: Candidate,
        job_description: JobDescription
    ) -> str:
        """
        Analyze experience level match.

        Args:
            candidate: Candidate profile
            job_description: Job requirements

        Returns:
            Experience match classification: "Exceeds", "Meets", or "Below"
        """
        candidate_years = candidate.total_years_experience or 0
        min_required = job_description.min_years_experience or 0
        max_required = job_description.max_years_experience

        if candidate_years >= (min_required + 3):  # 3+ years above minimum
            return "Exceeds"
        elif candidate_years >= min_required:
            if max_required and candidate_years > max_required:
                return "Exceeds"  # Too much experience might also be good
            return "Meets"
        else:
            return "Below"

    def _analyze_education_match(
        self,
        candidate: Candidate,
        job_description: JobDescription
    ) -> str:
        """
        Analyze education match.

        Args:
            candidate: Candidate profile
            job_description: Job requirements

        Returns:
            Education match classification: "Exceeds", "Meets", or "Below"
        """
        if not candidate.education or not job_description.education_requirements:
            return "Meets"  # Default if no data

        candidate_degrees = [edu.degree.lower() for edu in candidate.education if edu.degree]

        # Simple matching for now - can be enhanced
        required_degrees = [req.lower() for req in job_description.education_requirements]

        degree_matches = []
        for req_degree in required_degrees:
            if any(req_degree in cand_degree for cand_degree in candidate_degrees):
                degree_matches.append(req_degree)

        if len(degree_matches) == len(required_degrees):
            return "Exceeds"  # All requirements met
        elif degree_matches:
            return "Meets"  # Some requirements met
        else:
            return "Below"  # None met

    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Create fallback analysis when matching fails."""
        logger.warning("Using fallback job matching analysis")
        return {
            "matched_required_skills": [],
            "missing_required_skills": [],
            "matched_preferred_skills": [],
            "additional_relevant_skills": [],
            "experience_match": "Meets",
            "education_match": "Meets",
            "similarity_score": 0.5,
            "matching_rationale": "Fallback analysis due to processing failure"
        }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the agent chain.

        Args:
            input_data: Dictionary containing candidate and job_description from previous stages

        Returns:
            Dictionary with matching analysis
        """
        logger.info("Job Matcher Agent - Processing candidate-job match")

        candidate = input_data.get("candidate")
        job_description_data = input_data.get("job_description")

        if not candidate:
            raise ValueError("No candidate data provided for job matching")

        if not job_description_data:
            raise ValueError("No job description provided for matching")

        # Convert job_description_data to JobDescription model if it's a dict
        if isinstance(job_description_data, dict):
            job_description = JobDescription(**job_description_data)
        else:
            job_description = job_description_data

        # Perform matching analysis
        match_analysis = self.analyze_job_match(candidate, job_description)

        # Prepare output for next agent
        output = {
            "resume_path": input_data.get("resume_path"),
            "candidate": candidate,
            "job_description": job_description,
            "match_analysis": match_analysis,
            "agent_3_success": True,
            "agent_3_metadata": {
                "required_skills_matched": len(match_analysis.get("matched_required_skills", [])),
                "preferred_skills_matched": len(match_analysis.get("matched_preferred_skills", [])),
                "similarity_score": match_analysis.get("similarity_score", 0)
            }
        }

        logger.info("Job Matcher Agent - Processing complete")
        return output

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method."""
        return self.process(input_data)
