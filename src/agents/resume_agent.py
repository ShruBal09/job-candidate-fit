"""
Resume parsing agent using Pydantic AI.
Extracts structured data from redacted resume text.
"""
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from pydantic_ai.settings import ModelSettings

from src.config import LLM_MODELS
from src.models import ParsedResume
from src.tools.parse_dates_and_duration import parse_dates_and_duration
from src.tools.nli_entailment import nli_entailment_tool
from dotenv import load_dotenv
import os

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

model = LLM_MODELS["resume_parser"]

# Load prompt from version-controlled file
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / model.prompt_version / "resume_parser_prompt.txt"
RESUME_PROMPT = PROMPT_PATH.read_text()


# Create Pydantic AI agent with structured output for resume parsing
resume_agent = Agent(
    model=model.name,
    output_type=ParsedResume,
    system_prompt=RESUME_PROMPT,
    tools=[
        Tool(
            parse_dates_and_duration,
            name="parse_dates_and_duration",
            description="Parse dates and compute duration in months and years.",
        ),
        Tool(
            nli_entailment_tool,
            name="nli_entailment_check",
            description="Check whether evidence entails a factual hypothesis.",
        ),
    ],
    model_settings=ModelSettings(temperature=model.temperature, seed=42),
    retries=3,
    output_retries=3
)


async def parse_resume(resume_text: str, candidate_id: str) -> ParsedResume:
    """
    Parse resume text into structured format.
    
    Args:
        resume_text: Resume text with PII removed
        candidate_id: Candidate's ID
        
    Returns:
        ParsedResume with all extracted data
    """
    try:
        result = await resume_agent.run(
            f"Parse this resume for candidate {candidate_id}:\n\n{resume_text}"
        )
        result=result.output
    except Exception as e:
        print("\n=== RESUME PARSER FAILURE ===")
        print("Error:", e)
        raise

    # Ensure ID is set
    result.candidate_id = candidate_id
    
    return result

