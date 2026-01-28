"""
Tool definition for computing overall score across multiple parameters
"""
from src.config import WEIGHTS

def compute_overall_fit_score(required_skill_score: float,
                              prefered_skill_score: float,
                              experience_score: float,
                              qualification_score: float,
                              seniority_score: float) -> dict:
    """
    Compute the overall candidate–job fit score by combining weighted scores across key evaluation dimensions.

    Args:
        required_skill_score: Score (0–100) representing how well the candidate matches required job skills
        prefered_skill_score: Score (0–100) representing how well the candidate matches preferred (nice-to-have) job skills
        experience_score: Score (0–100) representing overall experience fit (years and type combined)
        qualification_score: Score (0–100) representing education and other qualification match
        seniority_score: Score (0–100) representing seniority alignment between the candidate and the role

    Returns:
        {
            "overall_fit_score": float (0–100)
        }
    """
    overall = (
        required_skill_score * WEIGHTS.required_skills
        + prefered_skill_score * WEIGHTS.prefered_skills
        + experience_score * WEIGHTS.experience
        + qualification_score * WEIGHTS.qualification
        + seniority_score * WEIGHTS.seniority
    )
    return {
        "overall_fit_score": round(overall, 2)
    }