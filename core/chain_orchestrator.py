"""LangGraph-based orchestrator for the AI agent chain."""

import os
from typing import Dict, Any, TypedDict, List, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from loguru import logger
import yaml

from agents.document_parser import DocumentParserAgent
from agents.info_extractor import InfoExtractorAgent
from agents.job_matcher import JobMatcherAgent
from agents.scorer import ScorerAgent
from agents.report_generator import ReportGeneratorAgent
from models.candidate import Candidate
from models.job_description import JobDescription
from utils.file_handler import FileHandler
from utils.vector_store import VectorStoreManager


class ChainState(TypedDict):
    """State for the LangGraph agent chain."""
    # Input data
    resume_path: Optional[str]
    job_description_data: Optional[Dict[str, Any]]
    output_path: Optional[str]

    # Agent 1: Document Parser
    raw_text: Optional[str]
    cleaned_text: Optional[str]
    agent_1_success: bool
    agent_1_metadata: Optional[Dict[str, Any]]

    # Agent 2: Information Extractor
    extracted_data: Optional[Dict[str, Any]]
    candidate: Optional[Candidate]
    agent_2_success: bool
    agent_2_metadata: Optional[Dict[str, Any]]

    # Agent 3: Job Matcher
    job_description: Optional[JobDescription]
    match_analysis: Optional[Dict[str, Any]]
    agent_3_success: bool
    agent_3_metadata: Optional[Dict[str, Any]]

    # Agent 4: Scorer & Ranker
    scoring_result: Optional[Dict[str, Any]]
    evaluation_result: Optional[Any]  # EvaluationResult or List[EvaluationResult]
    agent_4_success: bool
    agent_4_metadata: Optional[Dict[str, Any]]

    # Agent 5: Report Generator
    report: Optional[Dict[str, Any]]
    agent_5_success: bool
    agent_5_metadata: Optional[Dict[str, Any]]

    # Final output
    saved_path: Optional[str]
    final_status: Optional[str]


@dataclass
class ChainConfig:
    """Configuration for the agent chain."""
    llm_config: Dict[str, Any]
    vector_config: Dict[str, Any]
    scoring_config: Dict[str, Any]
    report_config: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str) -> "ChainConfig":
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)

            return cls(
                llm_config=config_data.get("llm", {}),
                vector_config=config_data.get("vector_store", {}),
                scoring_config=config_data.get("scoring", {}),
                report_config=config_data.get("reporting", {})
            )
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls.get_default_config()

    @classmethod
    def get_default_config(cls) -> "ChainConfig":
        """Get default configuration."""
        return cls(
            llm_config={"model": "gpt-4-turbo-preview", "temperature": 0.1},
            vector_config={"model_name": "all-MiniLM-L6-v2"},
            scoring_config={
                "weights": {
                    "skills_match": 0.40,
                    "experience_relevance": 0.30,
                    "education_match": 0.15,
                    "cultural_fit": 0.15
                }
            },
            report_config={
                "output_format": "json",
                "include_reasoning": True,
                "generate_interview_questions": True
            }
        )


class ResumeScreeningChain:
    """
    LangGraph-based agent chain orchestrator for resume screening.

    Coordinates the flow between 5 AI agents:
    1. Document Parser - Clean and format resume text
    2. Information Extractor - Extract structured candidate data
    3. Job Matcher - Analyze candidate-job fit with semantic matching
    4. Scorer & Ranker - Score candidate with weighted criteria
    5. Report Generator - Create comprehensive hiring reports
    """

    def __init__(self, config: Optional[ChainConfig] = None):
        """
        Initialize the agent chain.

        Args:
            config: Configuration for the chain and agents
        """
        self.config = config or ChainConfig.get_default_config()
        self.file_handler = FileHandler()
        self.vector_store = VectorStoreManager(self.config.vector_config.get("model_name", "all-MiniLM-L6-v2"))

        # Initialize agents
        self.agent_1 = DocumentParserAgent(llm_config=self.config.llm_config)
        self.agent_2 = InfoExtractorAgent(llm_config=self.config.llm_config)
        self.agent_3 = JobMatcherAgent(
            llm_config=self.config.llm_config,
            vector_config=self.config.vector_config
        )
        self.agent_4 = ScorerAgent(
            llm_config=self.config.llm_config,
            scoring_config=self.config.scoring_config
        )
        self.agent_5 = ReportGeneratorAgent(
            llm_config=self.config.llm_config,
            report_config=self.config.report_config
        )

        # Build the LangGraph
        self.graph = self._build_graph()

        logger.info("Resume Screening Chain initialized with all agents")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""

        def agent_1_process(state: ChainState) -> ChainState:
            """Node for Agent 1: Document Parser."""
            logger.info("Executing Agent 1: Document Parser")

            try:
                # Read and validate input file
                resume_path = state.get("resume_path")
                if not resume_path:
                    raise ValueError("No resume path provided")

                # Read resume file
                raw_text = self.file_handler.read_resume(resume_path)

                # Process with Agent 1
                input_data = {
                    "resume_path": resume_path,
                    "raw_text": raw_text
                }

                result = self.agent_1.process(input_data)

                # Update state
                return {
                    **state,
                    "raw_text": raw_text,
                    "cleaned_text": result.get("cleaned_text"),
                    "agent_1_success": result.get("agent_1_success", False),
                    "agent_1_metadata": result.get("agent_1_metadata"),
                    "final_status": "agent_1_complete"
                }

            except Exception as e:
                logger.error(f"Agent 1 failed: {e}")
                return {
                    **state,
                    "agent_1_success": False,
                    "final_status": f"agent_1_failed: {str(e)}"
                }

        def agent_2_process(state: ChainState) -> ChainState:
            """Node for Agent 2: Information Extractor."""
            logger.info("Executing Agent 2: Information Extractor")

            try:
                if not state.get("agent_1_success", False):
                    raise ValueError("Agent 1 must succeed before Agent 2")

                cleaned_text = state.get("cleaned_text")
                if not cleaned_text:
                    raise ValueError("No cleaned text available")

                input_data = {
                    "resume_path": state.get("resume_path"),
                    "raw_text": state.get("raw_text"),
                    "cleaned_text": cleaned_text
                }

                result = self.agent_2.process(input_data)

                return {
                    **state,
                    "extracted_data": result.get("extracted_data"),
                    "candidate": result.get("candidate"),
                    "agent_2_success": result.get("agent_2_success", False),
                    "agent_2_metadata": result.get("agent_2_metadata"),
                    "final_status": "agent_2_complete"
                }

            except Exception as e:
                logger.error(f"Agent 2 failed: {e}")
                return {
                    **state,
                    "agent_2_success": False,
                    "final_status": f"agent_2_failed: {str(e)}"
                }

        def agent_3_process(state: ChainState) -> ChainState:
            """Node for Agent 3: Job Matcher."""
            logger.info("Executing Agent 3: Job Matcher")

            try:
                if not state.get("agent_2_success", False):
                    raise ValueError("Agent 2 must succeed before Agent 3")

                candidate = state.get("candidate")
                if not candidate:
                    raise ValueError("No candidate data available")

                # Convert job description data to model if needed
                job_description_data = state.get("job_description_data")
                if isinstance(job_description_data, dict):
                    job_description = JobDescription(**job_description_data)
                elif job_description_data:
                    job_description = job_description_data
                else:
                    raise ValueError("No job description provided")

                input_data = {
                    "resume_path": state.get("resume_path"),
                    "candidate": candidate,
                    "job_description": job_description
                }

                result = self.agent_3.process(input_data)

                return {
                    **state,
                    "job_description": result.get("job_description"),
                    "match_analysis": result.get("match_analysis"),
                    "agent_3_success": result.get("agent_3_success", False),
                    "agent_3_metadata": result.get("agent_3_metadata"),
                    "final_status": "agent_3_complete"
                }

            except Exception as e:
                logger.error(f"Agent 3 failed: {e}")
                return {
                    **state,
                    "agent_3_success": False,
                    "final_status": f"agent_3_failed: {str(e)}"
                }

        def agent_4_process(state: ChainState) -> ChainState:
            """Node for Agent 4: Scorer & Ranker."""
            logger.info("Executing Agent 4: Scorer & Ranker")

            try:
                if not state.get("agent_3_success", False):
                    raise ValueError("Agent 3 must succeed before Agent 4")

                candidate = state.get("candidate")
                job_description = state.get("job_description")
                match_analysis = state.get("match_analysis")

                if not all([candidate, job_description, match_analysis]):
                    raise ValueError("Missing required data for scoring")

                input_data = {
                    "resume_path": state.get("resume_path"),
                    "candidate": candidate,
                    "job_description": job_description,
                    "match_analysis": match_analysis
                }

                result = self.agent_4.process(input_data)

                return {
                    **state,
                    "scoring_result": result.get("scoring_result"),
                    "evaluation_result": result.get("evaluation_result"),
                    "agent_4_success": result.get("agent_4_success", False),
                    "agent_4_metadata": result.get("agent_4_metadata"),
                    "final_status": "agent_4_complete"
                }

            except Exception as e:
                logger.error(f"Agent 4 failed: {e}")
                return {
                    **state,
                    "agent_4_success": False,
                    "final_status": f"agent_4_failed: {str(e)}"
                }

        def agent_5_process(state: ChainState) -> ChainState:
            """Node for Agent 5: Report Generator."""
            logger.info("Executing Agent 5: Report Generator")

            try:
                if not state.get("agent_4_success", False):
                    raise ValueError("Agent 4 must succeed before Agent 5")

                evaluation_result = state.get("evaluation_result")
                job_description = state.get("job_description")
                output_path = state.get("output_path")

                if not evaluation_result:
                    raise ValueError("No evaluation result available")

                input_data = {
                    "evaluation_result": evaluation_result,
                    "job_description": job_description
                }

                if output_path:
                    input_data["output_path"] = output_path

                result = self.agent_5.process(input_data)

                return {
                    **state,
                    "report": result.get("report"),
                    "agent_5_success": result.get("agent_5_success", False),
                    "agent_5_metadata": result.get("agent_5_metadata"),
                    "saved_path": result.get("saved_path"),
                    "final_status": "chain_complete"
                }

            except Exception as e:
                logger.error(f"Agent 5 failed: {e}")
                return {
                    **state,
                    "agent_5_success": False,
                    "final_status": f"agent_5_failed: {str(e)}"
                }

        # Conditional routing function
        def should_continue(state: ChainState) -> str:
            """Determine next step based on current state."""
            if state.get("agent_1_success", False):
                if state.get("agent_2_success", False):
                    if state.get("agent_3_success", False):
                        if state.get("agent_4_success", False):
                            return "agent_5"
                        return "agent_4"
                    return "agent_3"
                return "agent_2"
            return END  # Stop if agent 1 fails

        # Create the graph
        workflow = StateGraph(ChainState)

        # Add nodes
        workflow.add_node("agent_1", agent_1_process)
        workflow.add_node("agent_2", agent_2_process)
        workflow.add_node("agent_3", agent_3_process)
        workflow.add_node("agent_4", agent_4_process)
        workflow.add_node("agent_5", agent_5_process)

        # Add edges (conditional flow)
        workflow.add_conditional_edges(
            "agent_1",
            should_continue,
            {
                "agent_2": "agent_2",
                END: END
            }
        )
        workflow.add_conditional_edges(
            "agent_2",
            should_continue,
            {
                "agent_3": "agent_3",
                END: END
            }
        )
        workflow.add_conditional_edges(
            "agent_3",
            should_continue,
            {
                "agent_4": "agent_4",
                END: END
            }
        )
        workflow.add_conditional_edges(
            "agent_4",
            should_continue,
            {
                "agent_5": "agent_5",
                END: END
            }
        )
        workflow.add_edge("agent_5", END)

        # Set entry point
        workflow.set_entry_point("agent_1")

        return workflow.compile()

    def process_resume(self, resume_path: str, job_description: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single resume through the entire agent chain.

        Args:
            resume_path: Path to the resume file
            job_description: Job description data
            output_path: Optional output path for report

        Returns:
            Final result dictionary
        """
        logger.info(f"Starting resume screening for: {resume_path}")

        initial_state = ChainState(
            resume_path=resume_path,
            job_description_data=job_description,
            output_path=output_path,
            raw_text=None,
            cleaned_text=None,
            extracted_data=None,
            candidate=None,
            match_analysis=None,
            scoring_result=None,
            evaluation_result=None,
            report=None,
            final_status="starting"
        )

        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)

            logger.info("Resume screening chain completed")
            return dict(final_state)

        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            return {
                **dict(initial_state),
                "final_status": f"chain_failed: {str(e)}",
                "error": str(e)
            }

    def process_batch_resumes(
        self,
        resume_paths: List[str],
        job_description: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple resumes and generate a batch report.

        Args:
            resume_paths: List of resume file paths
            job_description: Job description data
            output_path: Optional output path for batch report

        Returns:
            Final batch result dictionary
        """
        logger.info(f"Starting batch processing for {len(resume_paths)} resumes")

        evaluation_results = []
        processed_count = 0
        failed_count = 0

        for resume_path in resume_paths:
            try:
                result = self.process_resume(resume_path, job_description)
                if result.get("final_status") == "chain_complete":
                    evaluation_results.append(result["evaluation_result"])
                    processed_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process {resume_path}: {result.get('final_status')}")

            except Exception as e:
                logger.error(f"Error processing {resume_path}: {e}")
                failed_count += 1

        logger.info(f"Batch processing complete: {processed_count} processed, {failed_count} failed")

        # Generate batch report if output path provided
        if output_path and evaluation_results:
            try:
                job_title = job_description.get("job_title", "Unknown Position")
                report_input = {
                    "evaluation_result": evaluation_results,
                    "job_description": job_description,
                    "output_path": output_path
                }

                final_result = self.agent_5.process(report_input)

                return {
                    "batch_completed": True,
                    "total_resumes": len(resume_paths),
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "report_generated": True,
                    "saved_path": final_result.get("saved_path"),
                    "report": final_result.get("report")
                }

            except Exception as e:
                logger.error(f"Failed to generate batch report: {e}")

        return {
            "batch_completed": False,
            "total_resumes": len(resume_paths),
            "processed_count": processed_count,
            "failed_count": failed_count,
            "report_generated": False,
            "error": "Batch processing failed"
        }
