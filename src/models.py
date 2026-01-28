"""
Pydantic data models for the candidate-job matching system.
Provides type-safe schemas for all data structures.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from datetime import datetime

# ============================================================================
# RESUME PROCESSING MODELS
# ============================================================================

class PIIEntity(BaseModel):
    """
    Represents a detected PII entity with its location and type.
    """
    entity_type: str = Field(description="Type: NAME, EMAIL, PHONE, URL, ADDRESS")
    text: str = Field(description="The detected PII text")
    start_char: int = Field(description="Start position in text")
    end_char: int = Field(description="End position in text")
    confidence: float = Field(ge=0.0, le=100.0, description="Detection confidence")
    replacement: str = Field(default="[REDACTED]", description="Replacement text")


class CandidateDetail(BaseModel):
    """
    Candidate details for contact.
    """
    id: str = Field(description="Candidate's unique ID", default="")
    name: str = Field(description="Candidate's full name", default="")
    email: str = Field(description="Candidate's contact email", default="")
    phone: str = Field(description="Candidate's contact phone number", default="")
    url: List[str] = Field(description="Candidate's URL details like LinkedIn profile, github etc.", default=[])
    location: str = Field(description="Candidate's location as a combination of City, Country of residence", default="")


class RedacteddResume(BaseModel):
    """
    Resume after PII has been detected and removed.
    """
    id: str = Field(description="Candidate unique ID", default="")
    original_text: str = Field(description="Original document text")
    redacted_text: str = Field(description="PII-free text")
    pii_entities: List[PIIEntity] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.now)

# ============================================================================
# Evidence
# ============================================================================

class EntityEvidence(BaseModel):
    """
    Single supporting evidence snippet for an extracted entity.
    """
    evidence_text: str = Field(description="Exact text snippet from source document")
    llm_confidence: float = Field(ge=0.0, le=100.0, description="Model confidence in this extraction")

# ============================================================================
# RESUME PARSING MODELS
# ============================================================================

class Education(BaseModel):
    """
    Educational qualification entry.
    """
    institution: str = Field(description="University or institution name")
    degree: Optional[str] = Field(None, description="Degree type")
    field_of_study: Optional[str] = Field(None, description="Major/specialisation")
    graduation_year: Optional[int] = Field(None, description="Year graduated")
    evidence: EntityEvidence = Field(description="Exact text from resume as evidence of education")

class Experience(BaseModel):
    """
    Work experience entry.
    """
    company: str = Field(description="Company name")
    title: str = Field(description="Job/role title")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date or Present")
    duration_months: Optional[int] = Field(None, description="Duration in months")
    description: Optional[str] = Field(None, description="Role description")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")
    evidence: EntityEvidence = Field(description="Exact text from resume as evidence of experience")


class ParsedResume(BaseModel):
    """
    Structured resume data extracted from redacted text.
    """
    candidate_id: str = Field(description="Candidate's unique ID")
    summary: Optional[str] = Field(None, description="Professional summary or bio")
    skills: List[str] = Field(default_factory=list, description="All skills - technical, professional and soft")
    skills_evidence: Dict[str, EntityEvidence] = Field(default_factory=dict, description="skill -> Exact text from resume as evidence of that skill")
    education: List[Education] = Field(default_factory=list, description="Education history")
    experience: List[Experience] = Field(default_factory=list, description="Work experience history")
    qualifications: List[str] = Field(default_factory=list, description="Certifications and licenses")
    qualifications_evidence: Dict[str, EntityEvidence] = Field(default_factory=dict, description="qualifications -> Exact text from resume as evidence of that qualification")
    total_experience_years: Optional[float] = Field(None, description="Total years experience")
    total_experience_evidence: EntityEvidence = Field(description="Exact text from resume as evidence of total experience")
    llm_confidence_overall: float = Field(ge=0.0, le=100.0, description="Overall parsing confidence")

# ============================================================================
# JOB DESCRIPTION MODELS
# ============================================================================

class ParsedJob(BaseModel):
    """
    Structured job description data.
    """
    job_id: str = Field(description="Unique job identifier")
    company: Optional[str] = Field(None, description="Company name")
    
    role_title: List[str] = Field(default_factory=list, description="Title for the role/ designation")
    role_title_evidence: List[EntityEvidence] = Field(description="Exact text from ad as evidence of role title")
    seniority: List[str] = Field(default_factory=list, description="Role seniority")
    seniority_evidence: List[EntityEvidence] = Field(description="Exact text from ad as evidence of seniority")
    industry: List[str] = Field(default_factory=list, description="Domain of role (not company)")
    industry_evidence: List[EntityEvidence] = Field(description="Exact text from ad as evidence of industry")
    
    required_skills: List[str] = Field(default_factory=list, description="Required, must have skills")
    required_skills_evidence: List[EntityEvidence] = Field(description="Exact text from ad as evidence of required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred, nice to have skills")
    preferred_skills_evidence: List[EntityEvidence] = Field(description="Exact text from ad as evidence of preferred skills")
    
    required_experience_years: Optional[float] = Field(None, description="Required experience in years")
    required_experience_years_evidence: Optional[EntityEvidence] = Field(description="Exact text from ad as evidence of experience required in years")
    required_experience_kind: Optional[str] = Field(None, description="Required experience in kind")
    required_experience_kind_evidence: Optional[EntityEvidence] = Field(description="Exact text from ad as evidence of kind of experience required")
    
    education_requirement: Optional[str] = Field(None, description="Education requirements")
    education_requirement_evidence: Optional[EntityEvidence] = Field(description="Exact text from ad as evidence of education requirement")
    other_qualifications: List[str] = Field(default_factory=list, description="Required Certification, licenses or other checks")
    other_qualifications_evidence: List[EntityEvidence] = Field(description="Exact text from ad as evidence of other qualifications")
    
    responsibilities: List[str] = Field(default_factory=list, description="Key job responsibilities")
    responsibilities_evidence: List[EntityEvidence] = Field(description="Exact text from ad as evidence of responsibilities")

    llm_confidence_overall: float = Field(ge=0.0, le=100.0, description="Overall parsing confidence")
    description: str = Field(description="Full job description")


# ============================================================================
# MATCHING AND SCORING MODELS
# ============================================================================

class SkillMatch(BaseModel):
    """Skill match analysis"""
    skill: str = Field(description="Job skill")
    result: Literal["match", "transferable", "missing"] = Field(description="Skill match kind")
    similarity: Optional[float] = Field(description="Match score", default=0.0)
    resume_evidence: Optional[EntityEvidence] = Field(description="Exact text from resume as evidence of candidate skill")
    job_evidence: Optional[EntityEvidence] = Field(description="Exact text from ad as evidence of job skill")


class ExperienceAssessment(BaseModel):
    """Years and kind of experience combined match."""
    years_score: float = Field(ge=0.0, le=100.0, description="Overall experience years match score")
    kind_score: float = Field(ge=0.0, le=100.0, description="Overall experience kind match score")
    experience_match_score: float = Field(ge=0.0, le=100.0, description="Overall experience match score")
    llm_confidence: float = Field(ge=0.0, le=100.0, description="Overall experience match confidence")
    evidence: Optional[EntityEvidence] = Field(description="Exact text from resume and ad as evidence of experience match")


class QualificationsAssessment(BaseModel):
    """Education and certifications match."""
    education_match_score: float = Field(ge=0.0, le=100.0, description="Overall education match score")
    other_qualifications_score: float = Field(ge=0.0, le=100.0, description="Overall other qualification match score")
    llm_confidence: float = Field(ge=0.0, le=100.0, description="Overall qualification match confidence")
    evidence: Optional[EntityEvidence] = Field(description="Exact text from resume and ad as evidence of qualifications match")


class SeniorityAssessment(BaseModel):
    """Seniority qualification assesment"""
    status: str = Field(description="One of under-qualified / qualified / over-qualified")  
    llm_confidence: float = Field(ge=0.0, le=100.0, description="Overall seniority match confidence")
    evidence: Optional[EntityEvidence] = Field(description="Exact text from resume and ad as evidence of seniority match")
    note: str = Field(description="Any additional notes")

class FitAnalysis(BaseModel):
    """
    Comprehensive candidate-job fit analysis.
    """
    candidate_id: str = Field(description="Candidate's unique ID")
    job_id: str = Field(description="Job identifier")
    overall_fit_score: float = Field(ge=0.0, le=100.0, description="Overall score")
    key_strengths: List[str] = Field(default_factory=list, description="Key strengths")
    
    # Summary
    recommendation: str = Field(description="Hiring recommendation - Progrss/Pass/Consider")
    recommendation_confidence: float = Field(ge=0.0, le=100.0, description="Confidence of recommendation score")
    
    # Component scores
    skill_match_score: float = Field(ge=0.0, le=100.0, description="Skill match score")
    experience_match_score: float = Field(ge=0.0, le=100.0, description="Experience score")
    education_match_score: float = Field(ge=0.0, le=100.0, description="Education score")
    
    # Detailed analysis
    skill_match: SkillMatch = Field(description="Skill matching details")
    experience_match: ExperienceAssessment = Field(description="Experience matching details")
    qualifications_match: QualificationsAssessment = Field(description="Qualifications matching details")
    seniority_match: SeniorityAssessment = Field(description="Seniority matching details")
    
    analysed_at: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


# ============================================================================
# REPORT MODELS
# ============================================================================

class SummaryGenerated(BaseModel):
    """
    AI generated summary of candidate fit
    """
    candidate_id: str = Field(description="Candidate identifier")
    job_id: str = Field(description="Job identifier")
    summary: str = Field(description="Summary of candidate fit for hiring manager")

class AnalysisReport(BaseModel):
    """
    Complete analysis report for export.
    """
    candidate_details: CandidateDetail = Field(description="Candidate's detail")
    job_id: str = Field(description="Job identifier")
    summary: SummaryGenerated = Field(description="AI generated summary")
    fit_analysis: FitAnalysis = Field(description="Fit analysis results")
    resume: ParsedResume = Field(description="Candidate resume data")
    job: ParsedJob = Field(description="Job description data")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation time")