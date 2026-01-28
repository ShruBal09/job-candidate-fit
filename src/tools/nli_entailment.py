"""
Tool definition for Dates parsing and duration calculation
"""
from __future__ import annotations

from typing import Dict
import torch
import logging

from src.utils.embedding_model import nli_model as model, nli_tokeniser as tokeniser


logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def nli_entailment_tool(claim: str, evidence: str)-> Dict[str, float]:
    """
    Args:
        claim: Short factual statement (e.g. "Candidate has skill python")
        evidence: Exact text snippet

    Returns:
        {
            "entailment_score": float between 0 and 1
        }
    """
    try:
        features = tokeniser(evidence, claim,  padding=True, truncation=True, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            scores = model(**features).logits
            probs = torch.softmax(scores, dim=-1)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        entailment = 0.0
        for item in probs:
                entailment = round(float(item[1]), 2)

        return {"entailment_score": entailment*100}
    except Exception as e:
        print(e)
        return None

    