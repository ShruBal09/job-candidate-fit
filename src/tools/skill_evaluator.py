"""
Tool definition for skills matching
"""
import re
import numpy as np
from typing import List, Dict
from src.utils.embedding_model import embed_model
from src.config import THRESHOLDS

def _norm(s: str) -> str:
    """Normalisation - Basic lower case and white space normalisation"""
    return re.sub(r"\s+", " ", s.lower().strip())

async def evaluate_single_skill(
    job_skill: str,
    candidate_skills: List[str],
) -> Dict:
    """
    Evaluate a single job-required skill against all candidate skills using
    lexical and semantic similarity.

    Args:
        job_skill: One skill explicitly required or preferred by the job description
        candidate_skills: List of all skills extracted from the candidate resume

    Returns:
        {
            "job_skill": str,
            "best_candidate_skill": str | None,
            "classification": str ("match" | "transferable" | "missing"),
            "similarity": float (0.0–1.0)
        }
    """
    job_norm = _norm(job_skill)
    cand_norm = [_norm(s) for s in candidate_skills]

    # Lexical
    if job_norm in cand_norm:
        return {
            "job_skill": job_skill,
            "best_candidate_skill": job_skill,
            "classification": "match",
            "similarity": 1.0,
        }

    # Semantic
    job_skill_embed = embed_model.encode(job_norm, normalize_embeddings=True)
    candidate_skill_embed = embed_model.encode(cand_norm, normalize_embeddings=True)

    sims = embed_model.similarity(job_skill_embed, candidate_skill_embed)
    max_sim_idx = int(np.argmax(sims))
    max_sim_val = float(sims[0][max_sim_idx].numpy())

    # Semantic match
    if max_sim_val>=THRESHOLDS.semantic_match:
        return {
            "job_skill": job_skill,
            "best_candidate_skill": candidate_skills[max_sim_idx],
            "classification": "match",
            "similarity": max_sim_val,
        }
    # Transferable skill match
    elif (max_sim_val>THRESHOLDS.skill_transferable_min) and (max_sim_val<THRESHOLDS.semantic_match):
        return {
            "job_skill": job_skill,
            "best_candidate_skill": candidate_skills[max_sim_idx],
            "classification": "transferable",
            "similarity": max_sim_val,
        }
    # Missing
    else:
        return {
            "job_skill": job_skill,
            "best_candidate_skill": candidate_skills[max_sim_idx],
            "classification": "missing",
            "similarity": 0,
        }