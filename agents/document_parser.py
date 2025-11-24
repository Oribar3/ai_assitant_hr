"""Agent 1: Document Parser - Cleans and formats resume text for extraction."""

from typing import Any
import json
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from utils.prompt_templates import PromptTemplates


class DocumentParserAgent:
    """
    First agent in the chain: Parses and cleans raw resume text.
    
    Role: Expert document parser specializing in resume extraction
    Task: Extract clean, structured text while preserving formatting and structure
    """

    def __init__(self, llm_config: dict = None):
        """
        Initialize the document parser agent.

        Args:
            llm_config: Configuration for the language model
        """
        self.llm_config = llm_config or {}
        self.llm = self._initialize_llm()
        self.prompt_template = PromptTemplates.DOCUMENT_PARSER
        logger.info("Document Parser Agent initialized")

    def _initialize_llm(self):
        """Initialize the language model with proper configuration."""
        default_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.1,  # Low temperature for consistent parsing
            "max_tokens": 2000,
        }
        config = {**default_config, **self.llm_config}
        return ChatOpenAI(**config)

    def parse_document(self, raw_text: str) -> str:
        """
        Parse and clean raw resume text.

        Args:
            raw_text: Raw text extracted from resume file

        Returns:
            Cleaned and structured text
        """
        logger.info("Starting document parsing")
        logger.debug(f"Input text length: {len(raw_text)} characters")

        try:
            # Format the prompt with the raw text
            prompt = PromptTemplates.format_prompt(
                self.prompt_template,
                resume_text=raw_text
            )

            # Create the prompt template for LangChain
            prompt_template = ChatPromptTemplate.from_template(prompt)

            # Create and run the chain
            chain = prompt_template | self.llm

            response = chain.invoke({})

            # Extract the content from the response
            cleaned_text = response.content.strip()

            logger.info("Document parsing completed successfully")
            logger.debug(f"Output text length: {len(cleaned_text)} characters")

            return cleaned_text

        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            # Return original text if parsing fails
            logger.warning("Returning original text due to parsing failure")
            return raw_text

    def validate_output(self, cleaned_text: str) -> bool:
        """
        Validate that the cleaned text meets basic requirements.

        Args:
            cleaned_text: The cleaned text output

        Returns:
            True if validation passes, False otherwise
        """
        if not cleaned_text or len(cleaned_text.strip()) < 50:
            logger.warning("Cleaned text is too short or empty")
            return False

        # Check for minimum content
        lines = cleaned_text.split('\n')
        content_lines = [line for line in lines if line.strip()]

        if len(content_lines) < 5:  # At least some basic sections
            logger.warning("Cleaned text has very few content lines")
            return False

        return True

    def process(self, input_data: dict) -> dict:
        """
        Main processing method for the agent chain.

        Args:
            input_data: Dictionary containing resume_path and other metadata

        Returns:
            Dictionary with cleaned_text and metadata
        """
        logger.info("Document Parser Agent - Processing resume")

        resume_path = input_data.get("resume_path")
        raw_text = input_data.get("raw_text")

        if not raw_text:
            raise ValueError("No raw text provided for parsing")

        # Parse and clean the text
        cleaned_text = self.parse_document(raw_text)

        # Validate output
        if not self.validate_output(cleaned_text):
            logger.warning("Output validation failed, using fallback cleaning")

            # Simple fallback: remove excessive whitespace and normalize
            cleaned_text = self._fallback_cleaning(raw_text)

        # Prepare output for next agent
        output = {
            "resume_path": resume_path,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "agent_1_success": True,
            "agent_1_metadata": {
                "original_length": len(raw_text),
                "cleaned_length": len(cleaned_text),
                "validation_passed": self.validate_output(cleaned_text)
            }
        }

        logger.info("Document Parser Agent - Processing complete")
        return output

    def _fallback_cleaning(self, text: str) -> str:
        """
        Fallback text cleaning method if AI parsing fails.

        Args:
            text: Raw text

        Returns:
            Basic cleaned text
        """
        logger.debug("Applying fallback text cleaning")

        # Basic cleaning steps
        import re

        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces to single

        # Remove special characters that might cause issues
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)  # Non-ASCII characters

        return cleaned.strip()

    # LangChain-compatible run method
    def run(self, input_data: dict) -> dict:
        """Alias for process method to match LangChain interface."""
        return self.process(input_data)
