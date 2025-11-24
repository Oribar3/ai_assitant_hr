"""Agent 4: Scorer & Ranker - Evaluates and scores candidates with recommendations."""

import json
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from models.candidate import Candidate
from models.job_description import JobDescription
from models.evaluation import ScoreBreakdown, EvaluationResult
from utils.prompt_templates import PromptTemplates


class ScorerAgent:
    """
    Fourth agent in the chain: Scores and ranks candidates.

    Role: Senior HR manager with extensive hiring expertise
    Task: Score candidate based on multiple criteria and provide hiring recommendation
    """

    def __init__(self, llm_config: dict = None, scoring_config: dict = None):
        """
        Initialize the scorer agent.

        Args:
            llm_config: Configuration for the language model
            scoring_config: Configuration for scoring weights and thresholds
        """
        self.llm_config = llm_config or {}
        self.scoring_config = scoring_config or self._get_default_scoring_config()
        self.llm = self._initialize_llm()
        self.prompt_template = PromptTemplates.SCORER
        logger.info("Scorer Agent initialized")

    def _get_default_scoring_config(self) -> Dict[str, Any]:
        """Get default scoring configuration."""
        return {
            "weights": {
                "skills_match": 0.40,
                "experience_relevance": 0.30,
                "education_match": 0.15,
                "cultural_fit": 0.15
            },
            "thresholds": {
                "excellent": 85,
                "good": 70,
                "moderate": 55,
                "low": 0
            }
        }

    def _initialize_llm(self):
        """Initialize the language model."""
        default_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.1,  # Low temperature for consistent scoring
            "max_tokens": 4000,
        }
        config = {**default_config, **self.llm_config}
        return ChatOpenAI(**config)

    def score_candidate(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        match_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score candidate based on multiple criteria.

        Args:
            candidate: Candidate profile
            job_description: Job requirements
            match_analysis: Analysis from Job Matcher Agent

        Returns:
            Dictionary containing detailed scoring and evaluation
        """
        logger.info("Starting candidate scoring")

        try:
            # Calculate individual component scores
            skills_score = self._calculate_skills_score(match_analysis)
            experience_score = self._calculate_experience_score(candidate, job_description, match_analysis)
            education_score = self._calculate_education_score(candidate, job_description, match_analysis)
            cultural_fit_score = self._calculate_cultural_fit_score(candidate, job_description, match_analysis)

            # Calculate overall weighted score
            overall_score = self._calculate_overall_score(
                skills_score, experience_score, education_score, cultural_fit_score
            )

            # Create score breakdown
            scores = {
                "skills_match": round(skills_score, 1),
                "experience_relevance": round(experience_score, 1),
                "education_match": round(education_score, 1),
                "cultural_fit": round(cultural_fit_score, 1),
                "overall_score": round(overall_score, 1)
            }

            # Get detailed evaluation using LLM
            detailed_evaluation = self._perform_detailed_evaluation(
                candidate, job_description, match_analysis, scores
            )

            # Combine results
            evaluation = {
                "scores": scores,
                "strengths": detailed_evaluation.get("strengths", []),
                "weaknesses": detailed_evaluation.get("weaknesses", []),
                "gaps": detailed_evaluation.get("gaps", []),
                "recommendation": detailed_evaluation.get("recommendation", "MAYBE"),
                "reasoning": detailed_evaluation.get("reasoning", "Evaluation completed"),
                "additional_insights": detailed_evaluation.get("additional_insights", "")
            }

            logger.info(f"Candidate scoring completed with overall score: {overall_score}")
            return evaluation

        except Exception as e:
            logger.error(f"Failed to score candidate: {e}")
            return self._create_fallback_evaluation()

    def _calculate_skills_score(self, match_analysis: Dict[str, Any]) -> float:
        """Calculate skills match score (0-100)."""
        matched_required = match_analysis.get("matched_required_skills", [])
        missing_required = match_analysis.get("missing_required_skills", [])
        matched_preferred = match_analysis.get("matched_preferred_skills", [])

        total_required = len(matched_required) + len(missing_required)

        if total_required == 0:
            return 100.0  # If no skills required, perfect match

        required_score = len(matched_required) / total_required

        # Bonus for preferred skills (up to 20% bonus)
        preferred_bonus = min(len(matched_preferred) * 0.1, 0.2)

        # Weights: 80% required skills, 20% preferred skills bonus
        skills_score = (required_score * 0.8) + (preferred_bonus * 0.2)

        return skills_score * 100

    def _calculate_experience_score(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        match_analysis: Dict[str, Any]
    ) -> float:
        """Calculate experience relevance score (0-100)."""
        experience_match = match_analysis.get("experience_match", "Meets")
        candidate_years = candidate.total_years_experience or 0
        min_required = job_description.min_years_experience or 0
        max_required = job_description.max_years_experience

        # Base score based on match classification
        base_score = {
            "Exceeds": 100,
            "Meets": 75,
            "Below": 35
        }.get(experience_match, 75)

        # Adjust based on experience level
        if experience_match == "Meets":
            # Slight bonus for exceeding required experience
            if candidate_years > min_required + 2:
                base_score += 10
        elif experience_match == "Exceeds":
            # Bonus for significantly more experience
            if max_required and candidate_years > max_required:
                base_score -= 10  # Penalty for too much experience (might be overqualified)
            elif candidate_years > min_required + 5:
                base_score += 5

        return min(max(base_score, 0), 100)

    def _calculate_education_score(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        match_analysis: Dict[str, Any]
    ) -> float:
        """Calculate education match score (0-100)."""
        education_match = match_analysis.get("education_match", "Meets")

        base_score = {
            "Exceeds": 100,
            "Meets": 80,
            "Below": 40
        }.get(education_match, 80)

        return base_score

    def _calculate_cultural_fit_score(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        match_analysis: Dict[str, Any]
    ) -> float:
        """Calculate cultural fit score (0-100)."""
        # This is a more subjective score based on company culture and soft skills

        # Base score from company culture alignment (if available)
        base_score = 70  # Default good cultural fit assumption

        # Adjust based on soft skills match
        soft_skills = candidate.soft_skills
        company_culture = job_description.company_culture

        # Simple heuristics for cultural fit
        if company_culture:
            culture_indicators = {
                "innovative": ["Innovation", "Creativity", "Problem Solving"],
                "collaborative": ["Teamwork", "Communication", "Leadership"],
                "analytical": ["Analytical Thinking", "Data Driven", "Attention to Detail"],
                "customer-focused": ["Customer Service", "Communication", "Empathy"]
            }

            culture_keywords = []
            for indicators in culture_indicators.values():
                culture_keywords.extend(indicators)

            matching_culture_skills = [
                skill for skill in soft_skills
                if any(keyword.lower() in skill.lower() for keyword in culture_keywords)
            ]

            if matching_culture_skills:
                base_score += len(matching_culture_skills) * 5
                base_score = min(base_score, 95)  # Cap at 95 for culture assessment

        return min(max(base_score, 0), 100)

    def _calculate_overall_score(
        self,
        skills_score: float,
        experience_score: float,
        education_score: float,
        cultural_fit_score: float
    ) -> float:
        """Calculate overall weighted score."""
        weights = self.scoring_config["weights"]

        overall = (
            skills_score * weights["skills_match"] +
            experience_score * weights["experience_relevance"] +
            education_score * weights["education_match"] +
            cultural_fit_score * weights["cultural_fit"]
        )

        # Normalize to ensure it's between 0-100
        return min(max(overall, 0), 100)

    def _perform_detailed_evaluation(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        match_analysis: Dict[str, Any],
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Use LLM to perform detailed evaluation and generate recommendations."""
        try:
            # Format the prompt
            prompt = PromptTemplates.format_prompt(
                self.prompt_template,
                candidate_profile=json.dumps({
                    "name": candidate.contact_info.name if candidate.contact_info else "Unknown",
                    "summary": candidate.summary,
                    "technical_skills": candidate.technical_skills,
                    "soft_skills": candidate.soft_skills,
                    "experience": candidate.total_years_experience,
                    "education": [edu.degree for edu in candidate.education] if candidate.education else []
                }, indent=2),
                job_description=json.dumps({
                    "title": job_description.job_title,
                    "required_skills": job_description.required_skills.technical_skills,
                    "min_experience": job_description.min_years_experience
                }, indent=2),
                match_analysis=json.dumps(match_analysis, indent=2),
                skills_weight=int(self.scoring_config["weights"]["skills_match"] * 100),
                experience_weight=int(self.scoring_config["weights"]["experience_relevance"] * 100),
                education_weight=int(self.scoring_config["weights"]["education_match"] * 100),
                cultural_fit_weight=int(self.scoring_config["weights"]["cultural_fit"] * 100)
            )

            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template(prompt)

            # Run evaluation
            chain = prompt_template | self.llm
            response = chain.invoke({})

            # Parse response
            response_text = response.content.strip()
            evaluation_result = self._parse_evaluation_response(response_text)

            # Merge with scores
            evaluation_result["scores"] = scores

            return evaluation_result

        except Exception as e:
            logger.error(f"Failed to perform detailed evaluation: {e}")
            return self._create_basic_evaluation(scores)

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM evaluation."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass

        # Extract key components manually
        return self._extract_evaluation_components(response_text)

    def _extract_evaluation_components(self, text: str) -> Dict[str, Any]:
        """Extract evaluation components from text if JSON parsing fails."""
        # Basic fallback extraction
        strengths = []
        weaknesses = []
        gaps = []

        lines = text.lower().split('\n')
        for i, line in enumerate(lines):
            if 'strength' in line:
                # Take next few lines as strengths
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('"'):
                        strengths.append(lines[j].strip('- ').capitalize())
            elif 'weakness' in line or 'concern' in line:
                # Take next few lines as weaknesses
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('"'):
                        weaknesses.append(lines[j].strip('- ').capitalize())

        recommendation = "MAYBE"  # Default
        if "strong hire" in text.lower():
            recommendation = "STRONG HIRE"
        elif "hire" in text.lower() and "strong" not in text.lower():
            recommendation = "HIRE"
        elif "pass" in text.lower() or "not recommended" in text.lower():
            recommendation = "PASS"

        return {
            "strengths": strengths[:5],  # Limit to 5
            "weaknesses": weaknesses[:4],  # Limit to 4
            "gaps": gaps,
            "recommendation": recommendation,
            "reasoning": text[:300] + "..." if len(text) > 300 else text
        }

    def _create_basic_evaluation(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Create basic evaluation when detailed analysis fails."""
        overall_score = scores["overall_score"]
        thresholds = self.scoring_config["thresholds"]

        if overall_score >= thresholds["excellent"]:
            recommendation = "STRONG HIRE"
        elif overall_score >= thresholds["good"]:
            recommendation = "HIRE"
        elif overall_score >= thresholds["moderate"]:
            recommendation = "MAYBE"
        else:
            recommendation = "PASS"

        return {
            "strengths": ["Good overall fit"],
            "weaknesses": ["Limited analysis available"],
            "gaps": [],
            "recommendation": recommendation,
            "reasoning": f"Score-based recommendation: {overall_score:.1f} overall score"
        }

    def _create_fallback_evaluation(self) -> Dict[str, Any]:
        """Create fallback evaluation when scoring fails."""
        logger.warning("Using fallback candidate evaluation")
        return {
            "scores": {
                "skills_match": 50,
                "experience_relevance": 50,
                "education_match": 50,
                "cultural_fit": 50,
                "overall_score": 50
            },
            "strengths": [],
            "weaknesses": [],
            "gaps": [],
            "recommendation": "MAYBE",
            "reasoning": "Fallback evaluation due to processing failure"
        }

    def create_evaluation_result(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        scoring_result: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Create an EvaluationResult Pydantic model from scoring results.

        Args:
            candidate: Candidate profile
            job_description: Job description
            scoring_result: Results from scoring

        Returns:
            EvaluationResult model instance
        """
        try:
            # Create ScoreBreakdown
            scores = scoring_result["scores"]
            score_breakdown = ScoreBreakdown(**scores)

            # Get candidate name
            candidate_name = (
                candidate.contact_info.name
                if candidate.contact_info and candidate.contact_info.name
                else "Unknown Candidate"
            )

            # Create evaluation result
            evaluation = EvaluationResult(
                candidate_name=candidate_name,
                job_title=job_description.job_title,
                scores=score_breakdown,
                match_analysis={},  # Will be populated from previous agent
                strengths=scoring_result.get("strengths", []),
                weaknesses=scoring_result.get("weaknesses", []),
                gaps=scoring_result.get("gaps", []),
                recommendation=scoring_result.get("recommendation", "MAYBE"),
                reasoning=scoring_result.get("reasoning", "Evaluation completed"),
                suggested_interview_questions=[]  # Will be added by Report Generator
            )

            return evaluation

        except Exception as e:
            logger.error(f"Failed to create EvaluationResult: {e}")
            return self._create_fallback_evaluation_result(candidate_name, job_description.job_title)

    def _create_fallback_evaluation_result(self, candidate_name: str, job_title: str) -> EvaluationResult:
        """Create fallback EvaluationResult."""
        return EvaluationResult(
            candidate_name=candidate_name,
            job_title=job_title,
            scores=ScoreBreakdown(
                skills_match=50, experience_relevance=50, education_match=50,
                cultural_fit=50, overall_score=50
            ),
            match_analysis={},
            strengths=[], weaknesses=[], gaps=[],
            recommendation="MAYBE",
            reasoning="Fallback evaluation due to processing failure"
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the agent chain.

        Args:
            input_data: Dictionary containing candidate, job_description, and match_analysis from previous agents

        Returns:
            Dictionary with scoring results
        """
        logger.info("Scorer Agent - Processing candidate evaluation")

        candidate = input_data.get("candidate")
        job_description = input_data.get("job_description")
        match_analysis = input_data.get("match_analysis")

        if not candidate:
            raise ValueError("No candidate data provided for scoring")

        if not job_description:
            raise ValueError("No job description provided for scoring")

        if not match_analysis:
            raise ValueError("No match analysis provided for scoring")

        # Perform comprehensive scoring
        scoring_result = self.score_candidate(candidate, job_description, match_analysis)

        # Create evaluation result model
        evaluation_result = self.create_evaluation_result(candidate, job_description, scoring_result)

        # Prepare output for next agent
        output = {
            "resume_path": input_data.get("resume_path"),
            "candidate": candidate,
            "job_description": job_description,
            "match_analysis": match_analysis,
            "scoring_result": scoring_result,
            "evaluation_result": evaluation_result,
            "agent_4_success": True,
            "agent_4_metadata": {
                "overall_score": scoring_result["scores"]["overall_score"],
                "recommendation": scoring_result["recommendation"],
                "confidence_level": self._assess_confidence(scoring_result)
            }
        }

        logger.info("Scorer Agent - Processing complete")
        return output

    def _assess_confidence(self, scoring_result: Dict[str, Any]) -> str:
        """Assess confidence in the evaluation based on data completeness."""
        scores = scoring_result["scores"]

        # High confidence if all scores are well-distributed
        score_range = max(scores.values()) - min(scores.values())
        if score_range > 30:  # Good differentiation between scores
            return "High"
        elif score_range > 15:
            return "Medium"
        else:
            return "Low"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method."""
        return self.process(input_data)
