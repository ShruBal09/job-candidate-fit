"""
Orchestrator coordinates the complete analysis pipeline.

Pipeline:
1. Load documents → text
2. Redact PII (resume only)
3. Parse resume 
4. Parse job
5. Match ad and resume → FitAnalysis
6. Generate summary
7. Return AnalysisReport
8. Regenerate summary with recruiter comments
"""
from datetime import datetime
from src.models import AnalysisReport, RedacteddResume, CandidateDetail, SummaryGenerated
from src.pii_redactor import PIIDetector
from src.utils.data_loader import load_document
from src.agents.resume_agent import parse_resume
from src.agents.job_agent import parse_job
from src.agents.matcher_agent import match_candidate_to_job
from src.agents.summary_agent import generate_summary
import json

class MatchingOrchestrator:
    """
    Orchestrates the complete candidate-job matching pipeline.
    """
    
    def __init__(self):
        """Initialise orchestrator"""
        self.pii_detector = PIIDetector()
        # Summary message thread
        self.message_thread = None
    
    async def analyse(
        self,
        resume_source: str,
        job_source: str,
        candidate_id: str = "",
        job_id: str = "",
        progress_callback=None
    )->AnalysisReport:
        """
        Run complete analysis pipeline.
        
        Args:
            resume_source: Path/URL to resume (PDF/HTML/TXT)
            job_source: Path/URL to job description
            candidate_id: Optional candidate ID (auto-generated if empty)
            job_id: Optional job ID (auto-generated if empty)
            progress_callback: Streamlit markdown object for status update
            
        Returns:
            AnalysisReport with complete analysis and recommendations
        """
        # Generate IDs if not provided
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not candidate_id:
            candidate_id = f"cand_{timestamp}"
        if not job_id:
            job_id = f"job_{timestamp}"
        
        print(f"Starting analysis: {candidate_id} → {job_id}")
        
        # ----------------------------------------------------------
        # Step 1: Load documents
        # ----------------------------------------------------------
        try:
            resume_text = load_document(resume_source)
            progress_callback(f"**Status:** Resume loaded ... **Now:** Loading job ad")

        except Exception as e:
            progress_callback(f"**Status:** Resume load failed")
            raise ValueError(f"Failed to load resume from {resume_source}: {e}")
        
        try:
            job_text = load_document(job_source)
            progress_callback(f"**Status:** Job loaded ... **Now:** Redacting resume")

        except Exception as e:
            progress_callback(f"**Status:** Job load failed")
            raise ValueError(f"Failed to load job from {job_source}: {e}")

        # ----------------------------------------------------------
        # Step 2: Redact PII from resume
        # ----------------------------------------------------------
        try:
            processed_resume, candidate_details = self.pii_detector.process_resume(resume_text, candidate_id)
            pii_count = len(processed_resume.pii_entities)
            print(f"Removed {pii_count} PII entities")
            print(f"Removed {processed_resume.pii_entities}")
            progress_callback("**Status:** Resume Redacted ... **Now:** Parsing resume")

        except Exception as e:
            print(f"PII redaction failed: {e}")
            progress_callback("**Status:** Resume Redaction failed")

            # Continue with original text if PII detection fails
            processed_resume = type('obj', (object,), {
                'redacted_text': resume_text,
                'id': candidate_id
            })
            processed_resume=RedacteddResume(
            id=candidate_id,
            original_text=resume_text,
            redacted_text=resume_text,
            pii_entities=[],
            processed_at=datetime.now()
            )
            candidate_details=CandidateDetail

        # ----------------------------------------------------------
        # Step 3: Parse resume
        # ----------------------------------------------------------
        try:
            parsed_resume = await parse_resume(processed_resume.redacted_text, candidate_id)
            progress_callback("**Status:** Resume parsed ... **Now:** Parsing job ad")

        except Exception as e:
            progress_callback("**Status:** Resume parse failed")
            raise Exception(f"Resume parsing failed: {e}")
        
        # ----------------------------------------------------------
        # Step 4: Parse job
        # ----------------------------------------------------------
        try:
            parsed_job = await parse_job(job_text, job_id)
            progress_callback("**Status:** Job ad parsed ... **Now:** Computing fit")

        except Exception as e:
            progress_callback("**Status:** Job ad parsing failed")
            raise Exception(f"Job parsing failed: {e}")

        # ----------------------------------------------------------
        # Step 5: Analyse fit
        # ----------------------------------------------------------
        try:
            fit_analysis = await match_candidate_to_job(parsed_resume, parsed_job)
            progress_callback("**Status:** Candidate fit analysed ... **Now:** Generating summary")

        except Exception as e:
            progress_callback("**Status:** Candidate fit analysis failed")
            raise Exception(f"Matching failed: {e}")
        
        # ----------------------------------------------------------
        # Step 6: Generate summary
        # ----------------------------------------------------------
        try:
            summary, message_thread = await generate_summary(fit_analysis, parsed_resume, parsed_job)
            self.message_thread = message_thread

        except Exception as e:
            raise Exception(f"Summary generation failed: {e}")

        # ----------------------------------------------------------
        # Step 7: Create complete report
        # ----------------------------------------------------------
        report = AnalysisReport(
            candidate_details=candidate_details,
            job_id=job_id,
            summary=summary,
            fit_analysis=fit_analysis,
            resume=parsed_resume,
            job=parsed_job,
            generated_at=datetime.now()
        )
        
        print(f"Summary generated!")
        return report


    async def regenerate_summary(
        self,
        report: AnalysisReport,
        recruiter_feedback: str
    )->SummaryGenerated:
        """
        Regenerate summary using recruiter feedback.
        
        Args:
            report: AnalysisReport from previous run
            recruiter_feedback: Recruiter's comments
            
        Returns:
            Summary regenerated by model
            
        """
        try:
            summary, message_thread = await generate_summary(
                fit_analysis=report.fit_analysis,
                resume=report.resume,
                job=report.job,
                recruiter_feedback=recruiter_feedback,
                message_history=self.message_thread
            )

            report.summary = summary        
            print(f"Summary regenerated!")
            self.message_thread = message_thread
            return summary

        except Exception as e:
            raise Exception(f"Summary generation failed: {e}")
