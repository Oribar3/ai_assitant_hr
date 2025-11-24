"""Agent 5: Report Generator - Creates comprehensive hiring reports and recommendations."""

import json
from typing import Dict, Any, List
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from models.evaluation import EvaluationResult, CandidateReport
from models.candidate import Candidate
from models.job_description import JobDescription
from utils.prompt_templates import PromptTemplates
from utils.file_handler import FileHandler


class ReportGeneratorAgent:
    """
    Fifth agent in the chain: Generates comprehensive hiring reports.

    Role: HR consultant creating executive summaries and hiring recommendations
    Task: Synthesize evaluation results into actionable reports with rankings and interview questions
    """

    def __init__(self, llm_config: dict = None, report_config: dict = None):
        """
        Initialize the report generator agent.

        Args:
            llm_config: Configuration for the language model
            report_config: Configuration for report generation
        """
        self.llm_config = llm_config or {}
        self.report_config = report_config or self._get_default_report_config()
        self.llm = self._initialize_llm()
        self.prompt_template = PromptTemplates.REPORT_GENERATOR
        self.file_handler = FileHandler()
        logger.info("Report Generator Agent initialized")

    def _get_default_report_config(self) -> Dict[str, Any]:
        """Get default report configuration."""
        return {
            "output_format": "json",  # json, csv, pdf
            "include_reasoning": True,
            "max_candidates_in_report": 20,
            "generate_interview_questions": True,
            "rank_by_overall_score": True
        }

    def _initialize_llm(self):
        """Initialize the language model."""
        default_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,  # Slightly higher temperature for creative report writing
            "max_tokens": 4000,
        }
        config = {**default_config, **self.llm_config}
        return ChatOpenAI(**config)

    def generate_report(self, evaluation_results: List[EvaluationResult], job_title: str) -> Dict[str, Any]:
        """
        Generate comprehensive hiring report from evaluation results.

        Args:
            evaluation_results: List of candidate evaluation results
            job_title: The job title being recruited for

        Returns:
            Dictionary containing the complete report
        """
        logger.info(f"Generating hiring report for {job_title} with {len(evaluation_results)} candidates")

        try:
            # Sort candidates by overall score (highest first)
            sorted_results = self._sort_candidates_by_score(evaluation_results)

            # Limit to max candidates if specified
            max_candidates = self.report_config["max_candidates_in_report"]
            if len(sorted_results) > max_candidates:
                sorted_results = sorted_results[:max_candidates]

            # Generate interview questions for top candidates
            if self.report_config["generate_interview_questions"]:
                sorted_results = self._generate_interview_questions(sorted_results)

            # Create executive summary and recommendations
            report_content = self._generate_report_content(sorted_results, job_title)

            # Compile final report
            report = {
                "job_title": job_title,
                "total_candidates_evaluated": len(evaluation_results),
                "report_date": datetime.now().isoformat(),
                "summary": report_content["summary"],
                "top_candidates": report_content["top_candidates"],
                "hiring_recommendations": report_content["hiring_recommendations"],
                "metadata": {
                    "candidates_included": len(sorted_results),
                    "highest_score": sorted_results[0].scores.overall_score if sorted_results else 0,
                    "average_score": self._calculate_average_score(evaluation_results),
                    "recommendation_distribution": self._analyze_recommendations(evaluation_results)
                }
            }

            logger.info("Report generation completed successfully")
            return report

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return self._create_fallback_report(job_title, evaluation_results)

    def _sort_candidates_by_score(self, evaluation_results: List[EvaluationResult]) -> List[EvaluationResult]:
        """Sort candidates by overall score (descending)."""
        return sorted(
            evaluation_results,
            key=lambda x: x.scores.overall_score,
            reverse=True
        )

    def _generate_interview_questions(self, sorted_results: List[EvaluationResult]) -> List[EvaluationResult]:
        """Generate tailored interview questions for top candidates."""
        logger.info("Generating interview questions for top candidates")

        # Generate questions for top 5 candidates (or all if fewer than 5)
        candidates_to_question = sorted_results[:5] if len(sorted_results) >= 5 else sorted_results

        try:
            for i, candidate in enumerate(candidates_to_question):
                questions = self._generate_candidate_questions(candidate, i + 1)
                candidate.suggested_interview_questions = questions

        except Exception as e:
            logger.error(f"Failed to generate interview questions: {e}")
            # Continue without questions if generation fails

        return sorted_results

    def _generate_candidate_questions(self, candidate: EvaluationResult, rank: int) -> List[str]:
        """Generate tailored interview questions for a specific candidate."""
        try:
            # Prepare prompt data
            context_data = {
                "job_title": candidate.job_title,
                "rank": rank,
                "overall_score": candidate.scores.overall_score,
                "strengths": candidate.strengths[:3],  # Top 3 strengths
                "gaps": candidate.gaps[:2] if candidate.gaps else []  # Top 2 gaps
            }

            prompt = f"""
Generate 4 tailored interview questions for a candidate ranked #{rank} for the position of {candidate.job_title}.

Candidate Profile:
- Overall Score: {candidate.scores.overall_score}/100
- Key Strengths: {', '.join(candidate.strengths[:3])}
- Areas of Concern: {', '.join(candidate.gaps[:2]) if candidate.gaps else 'None identified'}

Questions should be:
1. One technical question probing their strongest skill/technical competency
2. One behavioral question testing their experience and problem-solving
3. One question addressing a potential gap or area of growth
4. One cultural fit question assessing alignment with team values

Return ONLY a JSON array of 4 question strings, no additional text.
"""

            # Create prompt template and run
            prompt_template = ChatPromptTemplate.from_template(prompt)
            chain = prompt_template | self.llm
            response = chain.invoke({})

            # Parse response
            response_text = response.content.strip()
            questions = self._parse_question_response(response_text)

            return questions

        except Exception as e:
            logger.error(f"Failed to generate questions for rank {rank}: {e}")
            return self._get_fallback_questions(candidate)

    def _parse_question_response(self, response_text: str) -> List[str]:
        """Parse JSON array of questions from LLM response."""
        try:
            # Try direct JSON parsing
            questions = json.loads(response_text)
            if isinstance(questions, list) and len(questions) >= 3:
                return questions[:4]  # Take up to 4 questions
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown or text
        import re
        # Look for numbered list or bullet points
        lines = response_text.split('\n')
        questions = []

        for line in lines:
            line = line.strip()
            # Remove numbering: "1. ", "2. ", etc.
            line = re.sub(r'^\d+\.\s*', '', line)
            # Remove bullets: "- ", "* ", etc.
            line = re.sub(r'^[-*•]\s*', '', line)

            if line and len(line) > 10 and line.endswith('?'):
                questions.append(line)

        return questions[:4] if len(questions) >= 3 else self._get_generic_questions()

    def _get_fallback_questions(self, candidate: EvaluationResult) -> List[str]:
        """Get fallback questions when generation fails."""
        job_title = candidate.job_title.lower()

        # Generic but relevant questions based on job title
        questions = [
            f"Can you walk me through your experience with {job_title} roles and what you found most challenging?",
            "Tell me about a project where you had to solve a complex problem. What was your approach?",
            "How do you stay current with industry trends and best practices in your field?"
        ]

        # Add a culture fit question
        questions.append("Describe your ideal work environment and team collaboration style.")

        return questions

    def _get_generic_questions(self) -> List[str]:
        """Get completely generic fallback questions."""
        return [
            "Can you describe your most significant professional achievement?",
            "What attracted you to this role and our company?",
            "How do you approach problem-solving and decision-making?",
            "Tell me about a time you had to work with a difficult team member."
        ]

    def _generate_report_content(
        self,
        sorted_results: List[EvaluationResult],
        job_title: str
    ) -> Dict[str, Any]:
        """Generate the main report content including summary and recommendations."""
        try:
            # Prepare data for the LLM
            candidates_summary = []
            for i, candidate in enumerate(sorted_results[:10]):  # Top 10 for summary
                candidates_summary.append({
                    "rank": i + 1,
                    "name": candidate.candidate_name,
                    "score": candidate.scores.overall_score,
                    "recommendation": candidate.recommendation,
                    "key_strengths": candidate.strengths[:2]
                })

            evaluation_summary = {
                "job_title": job_title,
                "total_candidates": len(sorted_results),
                "evaluation_results": candidates_summary
            }

            # Format the report generation prompt
            prompt = PromptTemplates.format_prompt(
                self.prompt_template,
                evaluation_results=json.dumps(evaluation_summary, indent=2),
                job_title=job_title,
                total_candidates=len(sorted_results)
            )

            # Create prompt template and run
            prompt_template = ChatPromptTemplate.from_template(prompt)
            chain = prompt_template | self.llm
            response = chain.invoke({})

            # Parse response
            response_text = response.content.strip()
            report_data = self._parse_report_response(response_text)

            # Format top candidates
            top_candidates = self._format_top_candidates(sorted_results)

            return {
                "summary": report_data.get("summary", "Report generated successfully"),
                "top_candidates": top_candidates,
                "hiring_recommendations": report_data.get("hiring_recommendations", [])
            }

        except Exception as e:
            logger.error(f"Failed to generate report content: {e}")
            return {
                "summary": f"Analysis completed for {len(sorted_results)} candidates applying for {job_title}.",
                "top_candidates": self._format_top_candidates(sorted_results),
                "hiring_recommendations": ["Review top candidates based on overall scores"]
            }

    def _parse_report_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON report structure from LLM response."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON within response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass

        # Fallback: extract summary manually
        return {
            "summary": self._extract_summary_from_text(response_text),
            "hiring_recommendations": self._extract_recommendations_from_text(response_text)
        }

    def _extract_summary_from_text(self, text: str) -> str:
        """Extract summary from unstructured text."""
        # Look for lines that seem like summaries
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Look for summary-like sentences
            if len(line) > 50 and ('candidates' in line.lower() or 'analysis' in line.lower()):
                return line

        # Default fallback
        return "Comprehensive candidate evaluation completed."

    def _extract_recommendations_from_text(self, text: str) -> List[str]:
        """Extract recommendations from unstructured text."""
        recommendations = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # Look for recommendation-like lines (starting with action verbs)
            if line and any(line.lower().startswith(word) for word in ['consider', 'schedule', 'review', 'prioritize', 'recommend']):
                recommendations.append(line)

        if not recommendations:
            recommendations = [
                "Review scored candidates in order from highest to lowest overall score",
                "Consider scheduling interviews for candidates scoring 70+",
                "Have additional screening for candidates in the 55-70 range"
            ]

        return recommendations[:5]  # Limit to 5 recommendations

    def _format_top_candidates(self, sorted_results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """Format top candidates for the report."""
        formatted_candidates = []

        for i, candidate in enumerate(sorted_results):
            formatted_candidate = {
                "rank": i + 1,
                "name": candidate.candidate_name,
                "overall_score": candidate.scores.overall_score,
                "recommendation": candidate.recommendation,
                "key_highlights": candidate.strengths[:3],  # Top 3 strengths
                "concerns": candidate.weaknesses[:2] if candidate.weaknesses else [],  # Top 2 concerns
                "suggested_interview_questions": (
                    candidate.suggested_interview_questions[:4]
                    if candidate.suggested_interview_questions
                    else []
                )
            }
            formatted_candidates.append(formatted_candidate)

        return formatted_candidates

    def _calculate_average_score(self, evaluation_results: List[EvaluationResult]) -> float:
        """Calculate average overall score."""
        if not evaluation_results:
            return 0.0

        total_score = sum(candidate.scores.overall_score for candidate in evaluation_results)
        return round(total_score / len(evaluation_results), 1)

    def _analyze_recommendations(self, evaluation_results: List[EvaluationResult]) -> Dict[str, int]:
        """Analyze distribution of recommendations."""
        distribution = {
            "STRONG_HIRE": 0,
            "HIRE": 0,
            "MAYBE": 0,
            "PASS": 0
        }

        for result in evaluation_results:
            recommendation = result.recommendation
            if recommendation in distribution:
                distribution[recommendation] += 1

        return distribution

    def _create_fallback_report(self, job_title: str, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Create fallback report when generation fails."""
        logger.warning("Using fallback report generation")

        sorted_results = self._sort_candidates_by_score(evaluation_results)

        return {
            "job_title": job_title,
            "total_candidates_evaluated": len(evaluation_results),
            "report_date": datetime.now().isoformat(),
            "summary": f"Basic evaluation report for {job_title} position. {len(evaluation_results)} candidates evaluated.",
            "top_candidates": self._format_top_candidates(sorted_results[:10]),
            "hiring_recommendations": [
                "Review candidates based on overall scores",
                "Consider interviewing top performers",
                "Focus on candidates with strong skill matches"
            ],
            "metadata": {
                "candidates_included": len(sorted_results),
                "highest_score": sorted_results[0].scores.overall_score if sorted_results else 0,
                "average_score": self._calculate_average_score(evaluation_results),
                "recommendation_distribution": self._analyze_recommendations(evaluation_results)
            }
        }

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save the report to file in specified format.

        Args:
            report: The generated report data
            output_path: Path to save the report
        """
        format_type = self.report_config["output_format"].lower()

        if format_type == "json":
            self._save_json_report(report, output_path)
        elif format_type == "csv":
            self._save_csv_report(report, output_path)
        else:
            # Default to JSON
            self._save_json_report(report, output_path)

    def _save_json_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save report as JSON file."""
        try:
            json_path = output_path if output_path.endswith('.json') else f"{output_path}.json"
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(report, file, indent=2, ensure_ascii=False)

            logger.info(f"JSON report saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")

    def _save_csv_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save simplified report as CSV file."""
        try:
            import csv
            csv_path = output_path if output_path.endswith('.csv') else f"{output_path}.csv"

            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['rank', 'name', 'overall_score', 'recommendation', 'key_highlights']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for candidate in report.get('top_candidates', []):
                    writer.writerow({
                        'rank': candidate['rank'],
                        'name': candidate['name'],
                        'overall_score': candidate['overall_score'],
                        'recommendation': candidate['recommendation'],
                        'key_highlights': '; '.join(candidate.get('key_highlights', []))
                    })

            logger.info(f"CSV report saved to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV report: {e}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the agent chain.

        Args:
            input_data: Dictionary containing evaluation_result from Agent 4

        Returns:
            Dictionary with final report
        """
        logger.info("Report Generator Agent - Processing final report")

        evaluation_result = input_data.get("evaluation_result")
        job_description = input_data.get("job_description")

        if not evaluation_result:
            raise ValueError("No evaluation result provided for report generation")

        # For multiple candidates, evaluation_result should be a list
        # For single candidate, wrap in list
        if isinstance(evaluation_result, list):
            evaluation_results = evaluation_result
        else:
            evaluation_results = [evaluation_result]

        job_title = job_description.job_title if job_description else getattr(evaluation_result, 'job_title', 'Unknown Position')

        # Generate the report
        report = self.generate_report(evaluation_results, job_title)

        # Save report to file (optional)
        output_path = input_data.get("output_path")
        if output_path:
            self.save_report(report, output_path)

        # Prepare final output
        output = {
            "report": report,
            "job_title": job_title,
            "total_candidates": len(evaluation_results),
            "agent_5_success": True,
            "agent_5_metadata": {
                "report_format": self.report_config["output_format"],
                "include_reasoning": self.report_config["include_reasoning"],
                "interview_questions_generated": self.report_config["generate_interview_questions"]
            }
        }

        if output_path:
            output["saved_path"] = output_path

        logger.info("Report Generator Agent - Processing complete")
        return output

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method."""
        return self.process(input_data)
