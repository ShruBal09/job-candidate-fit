"""
Job description parsing agent using Pydantic AI.
Extracts structured requirements from job postings.
"""
from pathlib import Path
from pydantic_ai.tools import Tool
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent

from src.models import ParsedJob
from src.config import LLM_MODELS
from src.tools.nli_entailment import nli_entailment_tool
from dotenv import load_dotenv
import os

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

model = LLM_MODELS["matcher"]

# Load prompt from version-controlled file
PROMPT_VERSION = "v1"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / model.prompt_version / "job_parser_prompt.txt"
JOB_PROMPT = PROMPT_PATH.read_text()

# Create Pydantic AI agent with structured output for ad parsing
model = LLM_MODELS["job_parser"]
job_agent = Agent(
    model=model.name,
    output_type=ParsedJob,
    system_prompt=JOB_PROMPT,tools=[
        Tool(
            nli_entailment_tool,
            name="nli_entailment_check",
            description="Check whether evidence entails a factual job requirement.",
        )
    ],
    model_settings=ModelSettings(temperature=model.temperature, seed=42),
    retries=3,
    output_retries=3
)


async def parse_job(job_ad_text: str, job_id: str) -> ParsedJob:
    """
    Parse job description into structured format.
    
    Args:
        job_ad_text: Job description text (PII removed if any)
        job_id: Unique job identifier
        
    Returns:
        ParsedJob with all extracted requirements
    """
    try:
        result = await job_agent.run(
        f"Parse this job description with ID {job_id}:\n\n{job_ad_text}"
        )
        result=result.output
    except Exception as e:
        print("\n=== JOB PARSER FAILURE ===")
        print("Error:", e)
        raise

    # Ensure ID is set
    result.job_id = job_id
    
    return result