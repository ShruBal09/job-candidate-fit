"""
Summary generation agent using Pydantic AI.
Creates a fit summary for hiring manager.
"""
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from src.config import LLM_MODELS
from src.models import FitAnalysis, ParsedResume, ParsedJob, SummaryGenerated
from typing import Sequence

model = LLM_MODELS["summary"]

# Load prompt from version-controlled file
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / model.prompt_version / "summariser_prompt.txt"
SUMMARY_PROMPT = PROMPT_PATH.read_text()



# Create Pydantic AI agent with structured output for summary generation
summary_agent = Agent(
    model.name,
    output_type=SummaryGenerated,
    system_prompt=SUMMARY_PROMPT,
    model_settings=ModelSettings(temperature=model.temperature, seed=42),
    retries=3,
    output_retries=3
)


async def generate_summary(
    fit_analysis: FitAnalysis,
    resume: ParsedResume,
    job: ParsedJob,
    recruiter_feedback: str = None,
    message_history: Sequence = None
) -> SummaryGenerated:
    """
    Generate human-readable summary for hiring managers.
    
    Args:
        fit_analysis: Computed fit analysis
        resume: Parsed resume
        job: Parsed job description
        recruiter_feedback: Recruiter's feedback/ comments
        message_history: Agent's chat history to continue conversation
        
    Returns:
        SummaryGenerated object

    """    
    try:
        if recruiter_feedback is None:
            # Prepare context
            context = f"""
            Generate a hiring manager summary for this candidate-job match.

            CANDIDATE RESUME (parsed): {resume}
            
            JOB: {job}

            FIT ANALYSIS: {fit_analysis}

            Write a clear, actionable summary for the hiring manager.
            """
            
            result = await summary_agent.run(context)
        
        # Rerun with recruiter feedback
        else:
            context = f"""
            You are revising a hiring summary based on recruiter feedback.

            IMPORTANT RULES:
            - Fit analysis scores are informative and recruiter feedback is authoritative.
            - Do NOT recompute or invent facts.
            - Use recruiter feedback to refine interpretation and emphasis.
            - If recommendation changes, explain why.

            CANDIDATE RESUME (parsed):
            {resume}

            JOB DESCRIPTION:
            {job}

            FIT ANALYSIS:
            {fit_analysis}

            RECRUITER FEEDBACK:
            {recruiter_feedback}

            TASK:
            Produce a revised hiring manager summary.
            """

            result = await summary_agent.run(context, message_history=message_history)

        result_output = result.output
        result_message_thread = result.all_messages()

        # Ensure IDs
        result_output.candidate_id = resume.candidate_id
        result_output.job_id = job.job_id

    except Exception as e:
        print("\n=== SUMMARY FAILURE ===")
        print("Error:", e)
        raise

    return result_output, result_message_thread
