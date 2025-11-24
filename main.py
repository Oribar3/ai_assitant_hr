#!/usr/bin/env python3
"""CLI interface for the Resume Screening AI Agent Chain."""

import os
import json
import click
from pathlib import Path
from typing import List, Optional
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout

from core.chain_orchestrator import ResumeScreeningChain, ChainConfig
from utils.file_handler import FileHandler


console = Console()


@click.group()
@click.option('--config', '-c', default='config.yaml',
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """Resume Screening AI Agent Chain - Automated HR recruitment tool.

    Process resumes through a 5-agent AI pipeline for intelligent candidate evaluation.
    """
    # Setup logging
    if verbose:
        logger.remove()
        logger.add(console.print, level="DEBUG")
    else:
        logger.remove()
        logger.add(console.print, level="INFO")

    # Load configuration
    config_path = Path(config)
    if config_path.exists():
        ctx.obj = ChainConfig.from_yaml(str(config_path))
        logger.info(f"Loaded configuration from {config_path}")
    else:
        ctx.obj = ChainConfig.get_default_config()
        logger.warning(f"Configuration file not found: {config_path}. Using defaults.")

    ctx.obj.file_handler = FileHandler()


@cli.command()
@click.argument('resume_path', type=click.Path(exists=True))
@click.argument('job_description_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output path for evaluation report')
@click.option('--format', '-f', type=click.Choice(['json', 'pretty']),
              default='pretty', help='Output format')
@click.pass_context
def screen(ctx: click.Context, resume_path: str, job_description_path: str,
           output: Optional[str], format: str) -> None:
    """Screen a single resume against a job description.

    RESUME_PATH: Path to the resume file (PDF, DOCX, or TXT)
    JOB_DESCRIPTION_PATH: Path to the job description file (JSON or TXT)
    """
    console.print("[bold blue]🚀 Starting Resume Screening Pipeline[/bold blue]")
    console.print()

    try:
        # Load job description
        job_desc = load_job_description(job_description_path)

        # Initialize chain
        chain = ResumeScreeningChain(ctx.obj)

        # Process resume
        with console.status("[bold green]Processing resume through AI agent chain...") as status:
            result = chain.process_resume(resume_path, job_desc, output)

        # Display results
        if result.get("final_status") == "chain_complete":
            display_single_result(result, format, output)
        else:
            display_error(result.get("final_status", "Unknown error"))

    except Exception as e:
        logger.error(f"Screening failed: {e}")
        console.print(f"[bold red]❌ Screening failed: {e}[/bold red]")


@cli.command()
@click.argument('resume_directory', type=click.Path(exists=True))
@click.argument('job_description_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output path for batch evaluation report')
@click.option('--max-resumes', '-m', type=int, default=20,
              help='Maximum number of resumes to process')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'summary']),
              default='summary', help='Output format')
@click.pass_context
def batch(ctx: click.Context, resume_directory: str, job_description_path: str,
          output: str, max_resumes: int, format: str) -> None:
    """Process multiple resumes in batch mode.

    RESUME_DIRECTORY: Directory containing resume files
    JOB_DESCRIPTION_PATH: Path to the job description file (JSON or TXT)
    """
    console.print("[bold blue]🚀 Starting Batch Resume Screening[/bold blue]")
    console.print()

    try:
        # Load job description
        job_desc = load_job_description(job_description_path)

        # Find resume files
        resume_paths = find_resume_files(resume_directory, max_resumes)

        if not resume_paths:
            console.print("[bold red]❌ No resume files found in directory[/bold red]")
            return

        console.print(f"Found {len(resume_paths)} resume files to process")
        console.print()

        # Initialize chain
        chain = ResumeScreeningChain(ctx.obj)

        # Process batch
        with console.status("[bold green]Processing resumes through AI agent chain...") as status:
            result = chain.process_batch_resumes(resume_paths, job_desc, output)

        # Display batch results
        if result.get("batch_completed"):
            display_batch_result(result, format, output)
        else:
            display_error(f"Batch processing failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        console.print(f"[bold red]❌ Batch processing failed: {e}[/bold red]")


@cli.command()
@click.argument('job_description_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output path to save structured job description')
@click.pass_context
def parse_job(ctx: click.Context, job_description_path: str, output: Optional[str]) -> None:
    """Parse and structure a job description file.

    JOB_DESCRIPTION_PATH: Path to the job description file
    """
    console.print("[bold blue]📝 Parsing Job Description[/bold blue]")
    console.print()

    try:
        # Load and structure job description
        job_desc = load_job_description(job_description_path)

        # Display structured output
        console.print("[bold green]✅ Job Description Parsed Successfully[/bold green]")
        console.print()

        # Create formatted display
        table = Table(title="Job Requirements")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="yellow")

        table.add_row("Job Title", job_desc.get("job_title", "N/A"))
        table.add_row("Required Skills",
                     ", ".join(job_desc.get("required_skills", {}).get("technical_skills", [])))
        table.add_row("Preferred Skills",
                     ", ".join(job_desc.get("preferred_skills", {}).get("technical_skills", [])) if job_desc.get("preferred_skills") else "None")
        table.add_row("Min Experience", str(job_desc.get("min_years_experience", "Not specified")))
        table.add_row("Company", job_desc.get("company", "Not specified"))

        console.print(table)
        console.print()

        # Save if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(job_desc, f, indent=2)
            console.print(f"[bold green]📁 Saved structured job description to: {output}[/bold green]")

    except Exception as e:
        logger.error(f"Job description parsing failed: {e}")
        console.print(f"[bold red]❌ Job description parsing failed: {e}[/bold red]")


@cli.command()
@click.option('--path', '-p', type=click.Path(exists=True),
              help='Path to specific resume file to validate')
@click.option('--directory', '-d', type=click.Path(exists=True),
              help='Directory to scan for resume files')
@click.pass_context
def validate(ctx: click.Context, path: Optional[str], directory: Optional[str]) -> None:
    """Validate resume files for compatibility.

    Checks if files can be processed by the system.
    """
    console.print("[bold blue]🔍 Validating Resume Files[/bold blue]")
    console.print()

    if path and directory:
        console.print("[bold red]❌ Specify either --path or --directory, not both[/bold red]")
        return

    try:
        file_handler = ctx.obj.file_handler

        if path:
            # Validate single file
            is_valid, error_msg = file_handler.validate_file(path)
            if is_valid:
                console.print(f"[bold green]✅ Valid: {path}[/bold green]")
                console.print("   File can be processed by the system.")
            else:
                console.print(f"[bold red]❌ Invalid: {path}[/bold red]")
                console.print(f"   Error: {error_msg}")

        elif directory:
            # Validate directory
            resume_files = find_resume_files(directory, float('inf'))
            if not resume_files:
                console.print("[bold red]❌ No resume files found in directory[/bold red]")
                return

            valid_count = 0
            invalid_count = 0

            for resume_path in resume_files:
                is_valid, error_msg = file_handler.validate_file(resume_path)
                if is_valid:
                    console.print(f"[bold green]✅ Valid: {Path(resume_path).name}[/bold green]")
                    valid_count += 1
                else:
                    console.print(f"[bold red]❌ Invalid: {Path(resume_path).name}[/bold red]")
                    console.print(f"   Error: {error_msg}")
                    invalid_count += 1

            console.print()
            console.print(f"[bold blue]Summary: {valid_count} valid, {invalid_count} invalid files[/bold blue]")

        else:
            console.print("[bold red]❌ Specify either --path or --directory[/bold red]")
            console.print("   Use --help for more information")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        console.print(f"[bold red]❌ Validation failed: {e}[/bold red]")


def load_job_description(file_path: str) -> dict:
    """Load job description from file."""
    path = Path(file_path)

    if path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    else:
        # Assume text file
        with open(path, 'r') as f:
            content = f.read()

        # Try to extract JSON if present in text
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except:
            pass

        # Otherwise, create basic structure
        return {
            "job_title": "Unknown Position",
            "summary": content[:500],
            "required_skills": {
                "technical_skills": [],  # Would need AI extraction for better parsing
                "soft_skills": [],
                "certifications": [],
                "languages": []
            },
            "min_years_experience": None
        }


def find_resume_files(directory: str, max_files: int) -> List[str]:
    """Find resume files in directory."""
    directory_path = Path(directory)
    supported_extensions = {'.pdf', '.docx', '.txt', '.html'}

    resume_files = []
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            resume_files.append(str(file_path))
            if len(resume_files) >= max_files:
                break

    return resume_files


def display_single_result(result: dict, format: str, output_path: Optional[str]) -> None:
    """Display results for single resume screening."""
    if format == 'json':
        if output_path:
            console.print(f"[bold green]✅ Report saved to: {output_path}[/bold green]")
        else:
            console.print_json(data=result.get("report", {}))
        return

    # Pretty format
    evaluation_result = result.get("evaluation_result")
    if not evaluation_result:
        display_error("No evaluation result available")
        return

    console.print("[bold green]✅ Resume Screening Completed![/bold green]")
    console.print()

    # Overall score box
    score = evaluation_result.scores.overall_score
    recommendation = evaluation_result.recommendation

    score_panel = Panel.fit(
        f"[bold cyan]Score: {score}/100[/bold cyan]\n[bold yellow]Recommendation: {recommendation}[/bold yellow]",
        title="📊 Overall Assessment"
    )
    console.print(score_panel)
    console.print()

    # Score breakdown
    scores_table = Table(title="📈 Score Breakdown")
    scores_table.add_column("Component", style="cyan", width=20)
    scores_table.add_column("Score", style="yellow", justify="right", width=10)
    scores_table.add_column("Weight", style="magenta", justify="right", width=10)

    scores_table.add_row("Skills Match",
                        f"{evaluation_result.scores.skills_match}",
                        "40%")
    scores_table.add_row("Experience Relevance",
                        f"{evaluation_result.scores.experience_relevance}",
                        "30%")
    scores_table.add_row("Education Match",
                        f"{evaluation_result.scores.education_match}",
                        "15%")
    scores_table.add_row("Cultural Fit",
                        f"{evaluation_result.scores.cultural_fit}",
                        "15%")

    console.print(scores_table)
    console.print()

    # Strengths and weaknesses
    if evaluation_result.strengths:
        strengths_panel = Panel.fit(
            "\n".join(f"• {strength}" for strength in evaluation_result.strengths),
            title="💪 Key Strengths"
        )
        console.print(strengths_panel)

    if evaluation_result.weaknesses:
        weaknesses_panel = Panel.fit(
            "\n".join(f"• {concern}" for concern in evaluation_result.weaknesses),
            title="⚠️  Concerns"
        )
        console.print(weaknesses_panel)

    # Report saved message
    if output_path:
        console.print()
        console.print(f"[bold green]💾 Full report saved to: {output_path}[/bold green]")


def display_batch_result(result: dict, format: str, output_path: str) -> None:
    """Display results for batch resume processing."""
    console.print("[bold green]✅ Batch Processing Completed![/bold green]")
    console.print()

    processed = result.get("processed_count", 0)
    failed = result.get("failed_count", 0)
    total = result.get("total_resumes", 0)

    success_rate = (processed / total * 100) if total > 0 else 0

    stats_panel = Panel.fit(
        f"Processed: {processed}\n"
        f"Failed: {failed}\n"
        f"Total: {total}\n"
        f"Success Rate: {success_rate:.1f}%",
        title="📊 Batch Processing Summary"
    )
    console.print(stats_panel)
    console.print()

    if result.get("report_generated"):
        console.print(f"[bold green]📄 Batch report saved to: {output_path}[/bold green]")
        console.print()

        if format == "summary":
            display_batch_summary(result.get("report", {}))
    else:
        console.print("[bold red]❌ Report generation failed[/bold red]")


def display_batch_summary(report: dict) -> None:
    """Display summary of batch report."""
    if not report:
        return

    # Top candidates table
    top_candidates = report.get("top_candidates", [])[:10]

    if top_candidates:
        table = Table(title="🏆 Top Candidates Summary")
        table.add_column("Rank", style="cyan", width=8, justify="right")
        table.add_column("Name", style="yellow", width=25)
        table.add_column("Score", style="green", width=8, justify="right")
        table.add_column("Recommendation", style="magenta", width=15)
        table.add_column("Key Highlights", style="blue")

        for candidate in top_candidates:
            highlights = candidate.get("key_highlights", [])[:2]
            highlights_str = ", ".join(highlights)

            table.add_row(
                str(candidate["rank"]),
                candidate["name"][:24],
                f"{candidate['overall_score']}",
                candidate["recommendation"],
                highlights_str[:50] + "..." if len(highlights_str) > 50 else highlights_str
            )

        console.print(table)

    # Metadata
    metadata = report.get("metadata", {})
    console.print()
    console.print(f"[bold blue]Total Evaluated: {metadata.get('candidates_included', 0)}[/bold blue]")
    console.print(f"[bold blue]Highest Score: {metadata.get('highest_score', 'N/A')}[/bold blue]")


def display_error(error_message: str) -> None:
    """Display error message."""
    console.print(f"[bold red]❌ Error: {error_message}[/bold red]")


if __name__ == "__main__":
    cli()
