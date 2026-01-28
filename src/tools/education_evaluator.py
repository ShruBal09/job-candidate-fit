"""
Tool definition for education fit evaluation
"""
import re
import numpy as np
from typing import List, Dict, Optional
from src.utils.embedding_model import embed_model
from src.config import THRESHOLDS

def _norm(s: str) -> str:
    """Normalisation - Basic lower case and white space normalisation"""
    return re.sub(r"\s+", " ", s.lower().strip())

async def score_education_and_qualification(
    ad_required_qualification: Optional[str],
    candidate_qualification: List[str],
    education_or_qualification: str
) -> Dict:
    """
    Evaluate ONE job skill against ALL candidate skills.

    Args:
        ad_required_qualification: One Education or qualification required in the ad
        candidate_qualification: All candidate education and qualification to evaluate against
        education_or_qualification: Indicator - One of Education, Qualification
        
    Returns:
        {
            "{education_or_qualification}_required": ad requirement,
            "best_candidate_match": Closest match from candidate,
            "score": int (0-100) match score,
            "note": str
        }
    """
    # Missing requirement → neutral
    if not ad_required_qualification:
        return {f"{education_or_qualification}_required": None, "best_candidate_match": candidate_qualification[0], "score": 70.0, "note": f"Job does not specify {education_or_qualification} requirement."}

    # Candidate missing detail
    if not candidate_qualification:
        return {f"{education_or_qualification}_required": ad_required_qualification, "best_candidate_match": None, "score": 50.0, "note": f"Candidate {education_or_qualification} not specified."}

    ad_norm = _norm(ad_required_qualification)
    cand_norm = [_norm(e) for e in candidate_qualification]
    
    # Lexical
    if ad_norm in cand_norm:
        return {
            f"{education_or_qualification}_required": ad_required_qualification,
            "best_candidate_match": ad_required_qualification,
            "score": 100,
            "note": f"Job does not specify {education_or_qualification} requirement.",
        }

    # Semantic
    ad_education_embed = embed_model.encode(ad_norm, normalize_embeddings=True)
    candidate_education_embed = embed_model.encode(cand_norm, normalize_embeddings=True)

    sims = embed_model.similarity(ad_education_embed, candidate_education_embed)
    max_sim_idx = int(np.argmax(sims))
    max_sim_val = float(sims[0][max_sim_idx].numpy())

    # Semantic match
    if max_sim_val>=THRESHOLDS.semantic_match:
        return {
            f"{education_or_qualification}_required": ad_required_qualification,
            "best_candidate_match": candidate_qualification[max_sim_idx],
            "score": 100,
            "note": "Candidate meets requirement.",
        }
    # Partial match
    elif (max_sim_val>THRESHOLDS.education_partial_match_min) and (max_sim_val<THRESHOLDS.semantic_match):
        return {
            f"{education_or_qualification}_required": ad_required_qualification,
            "best_candidate_match": candidate_qualification[max_sim_idx],
            "score": 70,
            "note": "Candidate partially meets requirement.",
        }
    # Missing
    else:
        return {
            f"{education_or_qualification}_required": ad_required_qualification,
            "best_candidate_match": candidate_qualification[max_sim_idx],
            "score": 0,
            "note": "Candidate does not meet requirement.",
        }