"""
Configuration file - make changes for each run
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


# -----------------------------
# Model configuration
# -----------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    temperature: float
    prompt_version: str = "v1"

@dataclass(frozen=True)
class NLIModelConfig:
    name: str = "cross-encoder/nli-deberta-v3-base"

@dataclass(frozen=True)
class EmbedModelConfig:
    name: str = "Qwen/Qwen3-Embedding-0.6B"

# -----------------------------
# Thresholds & weights
# -----------------------------

@dataclass(frozen=True)
class Thresholds:
    semantic_match: float = 0.80
    skill_transferable_min: float = 0.50
    education_partial_match_min: float = 0.50
    experience_match_score: float = 0.80


@dataclass(frozen=True)
class Weights:
    required_skills: float = 0.30
    prefered_skills: float = 0.10
    experience: float = 0.30
    qualification: float = 0.15
    seniority: float = 0.15
    experience_years: float = 0.4
    experience_kind: float = 0.6


# -----------------------------
# Central registry
# -----------------------------

LLM_MODELS: Dict[str, ModelConfig] = {
    "resume_parser": ModelConfig(
        name="openai:gpt-4.1",
        temperature=0.3,
        prompt_version="v1"
    ),
    "job_parser": ModelConfig(
        name="openai:gpt-4.1",
        temperature=0.3,
        prompt_version="v1"
    ),
    "matcher": ModelConfig(
        name="openai:gpt-5-2025-08-07",
        temperature=0.5,
        prompt_version="v1"
    ),
    "summary": ModelConfig(
        name="openai:gpt-5-2025-08-07",
        temperature=0.5,
        prompt_version="v1"
    ),
}

THRESHOLDS = Thresholds()
WEIGHTS = Weights()
NLIModel= NLIModelConfig()
EMBEDModel = EmbedModelConfig()
