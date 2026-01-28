"""
Tool definition for experience (years and kind) fit evaluation
"""
from src.utils.embedding_model import embed_model
import numpy as np
from typing import List, Optional
import re
from src.config import WEIGHTS, THRESHOLDS


def _norm(s: str) -> str:
    """Normalisation - Basic lower case and white space normalisation"""
    return re.sub(r"\s+", " ", s.lower().strip())

def score_experience_years(candidate_years: Optional[float], required_years: Optional[float]) -> dict:
    """
    Evaluate candidate experience years against job experience requirement.

    Args:
        candidate_years: Total years of relevant experience extracted from the candidate resume
        required_years: Minimum years of experience required by the job description

    Returns:
        {
            "candidate_years": float | None,
            "required_years": float | None,
            "score": int (0–100) experience years match score,
            "note": str
        }
    """
    if required_years is None:
        return {"candidate_years": candidate_years, "required_years": required_years, "score": 70.0, "note": "Job years requirement not specified."}
    if candidate_years is None:
        return {"candidate_years": candidate_years, "required_years": required_years, "score": 50.0, "note": "Candidate years not specified."}
    if required_years <= 0:
        return {"candidate_years": candidate_years, "required_years": required_years, "score": 70.0, "note": "Non-positive required years treated as unspecified."}

    ratio = candidate_years / required_years
    score=ratio*100

    return {"candidate_years": candidate_years, "required_years": required_years, "score": score, "note": f"Candidate/required ratio={ratio:.2f}."}


def score_experience_kind(required_kind: Optional[str], candidate_experience_texts: List[str]) -> dict:
    """
    Evaluate the type or domain of experience required by the job against
    the candidate's past experience using lexical and semantic similarity.

    Args:
        required_kind: Description of one experience type required by the job
        candidate_experience_texts: List of candidate experience descriptions

    Returns:
        {
            "ad_required_kind": str | None,
            "candidate_matching_text": str | None,
            "score": int (0–100) experience kind match score,
            "similarity": float (0.0–1.0) semantic similarity score
        }
    """
    try:
        if not required_kind or not required_kind.strip():
            return {"ad_kind": None, "candidate_matching_text": candidate_experience_texts[0], "score": 70.0, "similarity": 0.0}
        if not candidate_experience_texts:
            return {"ad_kind": required_kind, "candidate_matching_text": None, "score": 30.0, "similarity": 0.0}
        if required_kind in candidate_experience_texts:
            return {"ad_kind": required_kind, "candidate_matching_text": required_kind, "score": 100.0, "similarity": 1.0}

        req = _norm(required_kind)
        cands_norm = [_norm(t) for t in candidate_experience_texts]
        req_embed = embed_model.encode(req, normalize_embeddings=True)
        cands_embed = embed_model.encode(cands_norm, normalize_embeddings=True)

        sims = embed_model.similarity(req_embed, cands_embed)
        max_sim_idx = int(np.argmax(sims))
        max_sim_val = float(sims[0][max_sim_idx].numpy())
        best_text = candidate_experience_texts[max_sim_idx] if max_sim_idx is not None else None

        # map similarity to score
        if max_sim_val >= THRESHOLDS.experience_match_score:
            score = 100.0
        else:
            score = max_sim_val*100

        return {"ad_required_kind": required_kind, "candidate_matching_text": best_text, "score": score, "similarity": float(max_sim_val)}
    except Exception as e:
        print("experience", e)

def combine_experience_scores(years_score: float, kind_score: float) -> dict:
    """
    Combine experience years score and experience type score into a single
    overall experience match score using a weighted approach.

    Args:
        years_score: Score (0–100) representing how well the candidate's
            total years of experience meet the job requirement
        kind_score: Score (0–100) representing how well the candidate's
            experience type or domain matches the job requirement

    Returns:
        {
            "years_score": float,
            "kind_score": float,
            "combined_score": float (0–100)
        }
    """
    combined = (years_score * WEIGHTS.experience_years) + (kind_score * WEIGHTS.experience_kind)
    return {"years_score": years_score, "kind_score": kind_score, "combined_score": float(round(combined, 2))}
