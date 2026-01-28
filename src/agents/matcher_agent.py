"""
Matching agent with semantic matching tools.

Flow:
1. Uses semantic_matcher tool to get skill matches (exact + transferable)
2. Uses semantic_matcher tool to score experience
3. LLM analyses seniority, strengths, overall fit
"""
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from pydantic_ai.settings import ModelSettings

from src.config import LLM_MODELS
from src.models import ParsedResume, ParsedJob, FitAnalysis
from src.tools.skill_evaluator import evaluate_single_skill
from src.tools.experience_evaluator import *
from src.tools.education_evaluator import score_education_and_qualification
from src.tools.overall_scoring import compute_overall_fit_score

model = LLM_MODELS["matcher"]

# Load prompt
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / model.prompt_version / "fit_analyser_prompt.txt"
MATCHER_PROMPT = PROMPT_PATH.read_text()


# Create Pydantic AI agent with structured output for candidate fit analysis
matcher_agent = Agent(
    model.name,
    output_type=FitAnalysis,
    system_prompt=MATCHER_PROMPT,
     tools=[
        Tool(evaluate_single_skill, name="evaluate_single_skill", description="Evaluate lexical and semantic match of ad and candidate skills. Returns classification: match | transferable | missing (with similarity)."),
        Tool(score_experience_years, name="score_experience_years", description="Deterministic score to evaluate candidate experience years match to ad requirement"),
        Tool(score_experience_kind, name="score_experience_kind", description="Lexical and semantic match to evaluate candidate experience field match to ad requirement"),
        Tool(combine_experience_scores, name="combine_experience_scores", description="Combine years + kind"),
        Tool(score_education_and_qualification, name="score_education_and_qualification", description="Use to evaluate lexical and semantic match of ad and candidate education and qualifications."),
        Tool(compute_overall_fit_score, name="compute_overall_fit_score", description="Weighted overall score combining required and prefered skills, experience, education, and qualifications."),
    ],
    model_settings=ModelSettings(temperature=model.temperature, seed=42),
    retries=3,
    output_retries=3
)


async def match_candidate_to_job(
    resume: ParsedResume,
    job: ParsedJob
) -> FitAnalysis:
    """
    Match candidate to job using semantic tools + LLM analysis.
    
    The agent will evaluate entity wise fit using tools and reasoning to compute overall fit and generate recommendation
    
    Args:
        resume: Parsed resume
        job: Parsed job description
        
    Returns:
        FitAnalysis with complete scoring
    """
    context = f"""
    Analyse this candidate-job match.

    CANDIDATE: {resume}

    JOB: {job}

    INSTRUCTIONS:
    Evaluate entity wise fit using tools and reasoning to compute overall fit and generate Accept/Consider/Reject recommendation

    Return complete FitAnalysis.
    """
    try:
        result = await matcher_agent.run(context)
        result=result.output
        # Ensure IDs
        result.candidate_id = resume.candidate_id
        result.job_id = job.job_id
    except Exception as e:
        print("\n=== MATCHER FAILURE ===")
        print("Error:", e)
        raise

    return result
