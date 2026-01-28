"""
Embedding models initialisation for use in agentic tools
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.config import NLIModel, EMBEDModel
from sentence_transformers import SentenceTransformer

nli_model = AutoModelForSequenceClassification.from_pretrained(NLIModel.name, dtype=torch.float32, low_cpu_mem_usage=False)
nli_tokeniser = AutoTokenizer.from_pretrained(NLIModel.name, use_fast=True)

embed_model = SentenceTransformer(EMBEDModel.name)
