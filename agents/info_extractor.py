"""Agent 2: Information Extractor - Extracts structured data from cleaned resume text."""

import json
from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from models.candidate import Candidate, ContactInfo, Experience, Education, Certification
from utils.prompt_templates import PromptTemplates


class InfoExtractorAgent:
    """
    Second agent in the chain: Extracts structured information from cleaned resume text.
    
    Role: HR data analyst expert in parsing structured information from resumes
    Task: Extract contact info, skills, experience, education into JSON format
    """

    def __init__(self, llm_config: dict = None):
        """
        Initialize the information extractor agent.

        Args:
            llm_config: Configuration for the language model
        """
        self.llm_config = llm_config or {}
        self.llm = self._initialize_llm()
        self.prompt_template = PromptTemplates.INFO_EXTRACTOR
        logger.info("Information Extractor Agent initialized")

    def _initialize_llm(self):
        """Initialize the language model with proper configuration."""
        default_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.1,  # Low temperature for structured extraction
            "max_tokens": 4000,  # Larger token limit for structured output
        }
        config = {**default_config, **self.llm_config}
        return ChatOpenAI(**config)

    def extract_information(self, cleaned_text: str) -> Dict[str, Any]:
        """
        Extract structured information from cleaned resume text.

        Args:
            cleaned_text: Cleaned resume text from Agent 1

        Returns:
            Dictionary containing extracted structured data
        """
        logger.info("Starting information extraction")
        logger.debug(f"Input text length: {len(cleaned_text)} characters")

        try:
            # Format the prompt
            prompt = PromptTemplates.format_prompt(
                self.prompt_template,
                resume_text=cleaned_text
            )

            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template(prompt)

            # Create and run chain
            chain = prompt_template | self.llm

            response = chain.invoke({})

            # Extract response content
            response_text = response.content.strip()

            logger.debug("Raw LLM response received, attempting to parse JSON")

            # Try to extract JSON from the response
            extracted_data = self._parse_json_response(response_text)

            # Validate and clean the extracted data
            cleaned_data = self.validate_and_clean_data(extracted_data)

            logger.info("Information extraction completed successfully")
            return cleaned_data

        except Exception as e:
            logger.error(f"Failed to extract information: {e}")
            # Return minimal fallback data
            return self._create_fallback_data(cleaned_text)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling various formats.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed JSON dictionary
        """
        # Try direct JSON parsing first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Look for JSON within markdown code blocks
        import re
        json_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_pattern, response_text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Look for JSON between { and }
        brace_pattern = r'\{.*\}'
        match = re.search(brace_pattern, response_text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: try to extract what looks like JSON
        logger.warning("Could not parse JSON, using fallback extraction")
        raise ValueError("Unable to parse JSON from LLM response")

    def validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted data to match expected structure.

        Args:
            data: Raw extracted data from LLM

        Returns:
            Cleaned and structured data
        """
        cleaned_data = {}

        # Extract contact info
        contact_data = data.get("contact_info", {})
        cleaned_data["contact_info"] = {
            "name": self._clean_string(contact_data.get("name", "")),
            "email": contact_data.get("email"),
            "phone": contact_data.get("phone"),
            "linkedin": contact_data.get("linkedin"),
            "location": contact_data.get("location"),
            "github": contact_data.get("github")
        }

        # Extract summary
        cleaned_data["summary"] = self._clean_string(data.get("summary", ""))

        # Extract and categorize skills
        skills_data = data.get("skills", {})
        cleaned_data["technical_skills"] = [
            self._clean_string(skill) for skill in skills_data.get("technical_skills", [])
        ] if isinstance(skills_data, dict) else []

        cleaned_data["soft_skills"] = [
            self._clean_string(skill) for skill in skills_data.get("soft_skills", [])
        ] if isinstance(skills_data, dict) else []

        cleaned_data["languages"] = [
            self._clean_string(skill) for skill in skills_data.get("languages", [])
        ] if isinstance(skills_data, dict) else []

        # Clean experience entries
        experience_data = data.get("experience", [])
        cleaned_data["experience"] = [
            self._clean_experience_entry(exp) for exp in experience_data
        ]

        # Clean education entries
        education_data = data.get("education", [])
        cleaned_data["education"] = [
            self._clean_education_entry(edu) for edu in education_data
        ]

        # Clean certification entries
        certifications_data = data.get("certifications", [])
        cleaned_data["certifications"] = [
            self._clean_certification_entry(cert) for cert in certifications_data
        ]

        # Calculate total years experience
        total_years = self._calculate_total_years(experience_data)
        cleaned_data["total_years_experience"] = total_years

        return cleaned_data

    def _clean_string(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize string values."""
        if not text:
            return None
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned if cleaned else None

    def _clean_experience_entry(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Clean experience entry data."""
        cleaned_exp = {
            "company": self._clean_string(exp.get("company", "")),
            "job_title": self._clean_string(exp.get("job_title", "")),
            "start_date": exp.get("start_date"),
            "end_date": exp.get("end_date"),
            "duration_months": exp.get("duration_months"),
            "description": self._clean_string(exp.get("description", "")),
            "achievements": [self._clean_string(a) for a in exp.get("achievements", []) if a],
            "technologies": [self._clean_string(t) for t in exp.get("technologies", []) if t]
        }
        return {k: v for k, v in cleaned_exp.items() if v is not None and v != []}

    def _clean_education_entry(self, edu: Dict[str, Any]) -> Dict[str, Any]:
        """Clean education entry data."""
        cleaned_edu = {
            "institution": self._clean_string(edu.get("institution", "")),
            "degree": self._clean_string(edu.get("degree", "")),
            "field_of_study": self._clean_string(edu.get("field_of_study")),
            "graduation_year": edu.get("graduation_year"),
            "gpa": edu.get("gpa"),
            "honors": self._clean_string(edu.get("honors"))
        }
        return {k: v for k, v in cleaned_edu.items() if v is not None}

    def _clean_certification_entry(self, cert: Dict[str, Any]) -> Dict[str, Any]:
        """Clean certification entry data."""
        cleaned_cert = {
            "name": self._clean_string(cert.get("name", "")),
            "issuing_organization": self._clean_string(cert.get("issuing_organization", "")),
            "issue_date": cert.get("issue_date"),
            "expiry_date": cert.get("expiry_date"),
            "credential_id": cert.get("credential_id")
        }
        return {k: v for k, v in cleaned_cert.items() if v is not None}

    def _calculate_total_years(self, experience_data: list) -> Optional[float]:
        """Calculate total years of professional experience."""
        try:
            total_months = 0
            for exp in experience_data:
                months = exp.get("duration_months")
                if months and isinstance(months, (int, float)):
                    total_months += months

            if total_months > 0:
                return round(total_months / 12, 1)
        except Exception as e:
            logger.debug(f"Failed to calculate total years: {e}")

        return None

    def _create_fallback_data(self, text: str) -> Dict[str, Any]:
        """Create minimal fallback data when extraction fails."""
        logger.warning("Using fallback data extraction")

        # Very basic extraction from first few lines
        lines = text.split('\n')[:10]  # First 10 lines only
        name_guess = None
        for line in lines:
            line = line.strip()
            if line and len(line.split()) <= 5:  # Likely a name line
                name_guess = line
                break

        return {
            "contact_info": {
                "name": name_guess or "Unknown",
                "email": None,
                "phone": None,
                "linkedin": None,
                "location": None,
                "github": None
            },
            "summary": None,
            "technical_skills": [],
            "soft_skills": [],
            "languages": [],
            "experience": [],
            "education": [],
            "certifications": [],
            "total_years_experience": None
        }

    def create_candidate_model(self, extracted_data: Dict[str, Any]) -> Candidate:
        """
        Create a Candidate Pydantic model from extracted data.

        Args:
            extracted_data: Structured data from extraction

        Returns:
            Candidate model instance
        """
        try:
            # Combine all skills for the general skills field
            all_skills = (
                extracted_data.get("technical_skills", []) +
                extracted_data.get("soft_skills", []) +
                extracted_data.get("languages", [])
            )

            candidate_data = {
                "contact_info": extracted_data.get("contact_info", {}),
                "summary": extracted_data.get("summary"),
                "skills": all_skills,
                "technical_skills": extracted_data.get("technical_skills", []),
                "soft_skills": extracted_data.get("soft_skills", []),
                "languages": extracted_data.get("languages", []),
                "experience": extracted_data.get("experience", []),
                "education": extracted_data.get("education", []),
                "certifications": extracted_data.get("certifications", []),
                "total_years_experience": extracted_data.get("total_years_experience"),
                "raw_text": None  # We'll set this from the input later
            }

            return Candidate(**candidate_data)
        except Exception as e:
            logger.error(f"Failed to create Candidate model: {e}")
            # Return minimal valid candidate
            return Candidate(
                contact_info={"name": "Error in parsing"},
                skills=[],
                technical_skills=[],
                soft_skills=[],
                languages=[],
                experience=[],
                education=[],
                certifications=[]
            )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the agent chain.

        Args:
            input_data: Dictionary containing cleaned_text from Agent 1

        Returns:
            Dictionary with extracted candidate data
        """
        logger.info("Information Extractor Agent - Processing resume")

        cleaned_text = input_data.get("cleaned_text")
        if not cleaned_text:
            raise ValueError("No cleaned text provided for information extraction")

        # Extract structured information
        extracted_data = self.extract_information(cleaned_text)

        # Create Pydantic model
        candidate = self.create_candidate_model(extracted_data)

        # Prepare output for next agent
        output = {
            "resume_path": input_data.get("resume_path"),
            "raw_text": input_data.get("raw_text"),
            "cleaned_text": cleaned_text,
            "extracted_data": extracted_data,
            "candidate": candidate,
            "agent_2_success": True,
            "agent_2_metadata": {
                "skills_extracted": len(extracted_data.get("technical_skills", [])),
                "experience_entries": len(extracted_data.get("experience", [])),
                "education_entries": len(extracted_data.get("education", []))
            }
        }

        logger.info("Information Extractor Agent - Processing complete")
        return output

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method."""
        return self.process(input_data)
