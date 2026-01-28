"""
Simple PII Detection using GLiNER.
Detects: NAME, EMAIL, PHONE, URL, LOCATION
"""
import re
from typing import List, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
from src.models import PIIEntity, CandidateDetail, RedacteddResume


class PIIDetector:
    """
    Simple PII detector using GLiNER model.
    
    Just loads model and detects PII - nothing fancy.
    """
    
    MODEL_NAME = "lakshyakh93/deberta_finetuned_pii"
    
    def __init__(self):
        """Load the model."""
        device="cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME, dtype=torch.float32)
        model.to(device)

        self.ner = hf_pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device
        )
        
        # Regex patterns for structured PII
        self.email_pattern = re.compile(
            r"""
            \b                              # Word boundary
            [A-Za-z0-9._%+-]+               # Local part (username)
            @                               # @ symbol
            [A-Za-z0-9.-]+                  # Domain name
            \.                              # Dot before TLD
            [A-Za-z]{2,}                   # Top-level domain (2+ letters)
            \b                              # Word boundary
            """,
            re.VERBOSE
        )

        self.phone_pattern = re.compile(
            r"""
            \b                              # Word boundary
            (?:\+?\d{1,3}[\s-]?)?           # Optional country code (+1, +61, +44)
            (?:                             # Optional area code
                \(?\d{2,4}\)?               # Area code with optional parentheses
                [\s-]?                      # Optional separator
            )?
            \d{3,4}                         # First local number block
            [\s-]?                          # Optional separator
            \d{3,4}                         # Second local number block
            \b                              # Word boundary
            """,
            re.VERBOSE
        )

        self.url_pattern = re.compile(
            r"""
            (?<!@)                               # Not part of an email
            \b
            (
                (?:https?://|www\.)              # Scheme OR www
                [a-z0-9-]+(\.[a-z0-9-]+)+         # Domain
                (?:/[^\s<>"']*)?                 # Optional path
                |
                [a-z0-9-]+(\.[a-z0-9-]+)+         # Bare domain
                /[^\s<>"']+                      # BUT must have a path
            )
            \b
            """,
            re.IGNORECASE | re.VERBOSE
        )

        
    def _normalise_label(self, label: str) -> str:
        """Map model labels to our types."""
        mapping = {
            "PERSON": "NAME",
            "FIRSTNAME": "NAME",
            "LASTNAME": "NAME",
            "NAME": "NAME",
            "EMAIL": "EMAIL",
            "PHONE_NUM": "PHONE",
            "URL": "URL",
            "ADDRESS": "LOCATION",
            "LOCATION": "LOCATION",
            "LOC": "LOCATION",
            "GPE": "LOCATION"
        }
        return mapping.get(label.upper(), label.upper())

    def _detect_with_regex(self, text: str) -> List[PIIEntity]:
        """
        Detect structured PII using regex.
        """
        entities = []

        patterns = [
            ("EMAIL", self.email_pattern),
            ("PHONE", self.phone_pattern),
            ("URL", self.url_pattern),
        ]

        for entity_type, pattern in patterns:
            for match in pattern.finditer(text):
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=1.0,  # deterministic
                    replacement=f"[{entity_type}]"
                ))

        return entities
    
    def detect_pii(self, text: str) -> List[PIIEntity]:
        """
        Detect PII entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of PIIEntity objects
        """        
        pii_entities = []
        
        # Regex-based detection (EMAIL, PHONE, URL)
        regex_entities = self._detect_with_regex(text)
        pii_entities.extend(regex_entities)

        # Track occupied spans
        occupied_spans = [
            (ent.start_char, ent.end_char) for ent in regex_entities
        ]

        # NER-based detection (NAME, LOCATION)
        entities = self.ner(text)
        current_entity = None  # buffer for merging spans

        for ent in entities:
            normalised = self._normalise_label(ent["entity_group"])
            if normalised not in {"NAME", "LOCATION"}:
                continue
            start, end = ent["start"], ent["end"]

            # Skip if overlaps with regex-detected span
            if any(not (end <= s or start >= e) for s, e in occupied_spans):
                continue

            # If starting a new entity OR entity type changes OR span is non-contiguous (with space)
            if (
                current_entity is None
                or current_entity["entity_type"] != normalised
                or start > current_entity["end_char"] + 1
            ):
                # Flush previous entity - if entity changes
                if current_entity is not None:
                    pii_entities.append(PIIEntity(**current_entity))
                    occupied_spans.append((current_entity["start_char"], current_entity["end_char"]))

                # Start new entity
                current_entity = {
                    "entity_type": normalised,
                    "entity_group": ent["entity_group"],
                    "text": ent["word"],
                    "start_char": start,
                    "end_char": end,
                    "confidence": float(ent["score"]),
                    "replacement": f"[{normalised}]",
                }

            else:
                # Merge with current entity

                # If same sub type ex. First name - First name, append as is, else First name - Last name, add space
                if ent["entity_group"] == current_entity["entity_group"]:
                    text = ent["word"]
                else:
                    text = " " + ent["word"]
                current_entity["text"] += text
                current_entity["entity_group"] = ent["entity_group"]
                current_entity["end_char"] = end
                current_entity["confidence"] = max(
                    current_entity["confidence"],
                    float(ent["score"])
                )

        # Flush final entity
        if current_entity is not None:
            pii_entities.append(PIIEntity(**current_entity))
            occupied_spans.append((current_entity["start_char"], current_entity["end_char"]))

        return pii_entities
    
    def process_resume(self, text: str, candidate_id: str = "") -> Tuple[RedacteddResume, CandidateDetail]:
        """
        Remove PII from resume and process contact details
        
        Args:
            text: Original resume text
            candidate_id: Candidate ID
            
        Returns:
            RedacteddResume with PII removed
            CandidateDetail with contact info
        """
        # Extract PII
        pii_entities = self.detect_pii(text)
        
        # Redact resume
        # Replace PII from end to start to preserve positions while replacing
        redacted_text = text
        for entity in sorted(pii_entities, key=lambda e: e.start_char, reverse=True):
            redacted_text = (
                redacted_text[:entity.start_char] +
                entity.replacement +
                redacted_text[entity.end_char:]
            )

        resume_id = candidate_id or f"cand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        processed_resume=RedacteddResume(
            id=resume_id,
            original_text=text,
            redacted_text=redacted_text,
            pii_entities=pii_entities,
            processed_at=datetime.now()
        )

        # Process candidate details
        detail = CandidateDetail(id=resume_id)
        
        # Extract first occurrence of each type
        for entity in pii_entities:
            if entity.entity_type == "NAME" and not detail.name:
                detail.name = entity.text
            elif entity.entity_type == "EMAIL" and not detail.email:
                detail.email = entity.text
            elif entity.entity_type == "PHONE" and not detail.phone:
                detail.phone = entity.text
            elif entity.entity_type == "LOCATION" and not detail.location:
                detail.location = entity.text
            elif entity.entity_type == "URL" and not detail.url:
                detail.url = [entity.text]
            elif entity.entity_type == "URL":
                detail.url.append(entity.text)
        return (processed_resume, detail)