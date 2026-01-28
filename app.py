"""
Streamlit UI that accepts ads and resumes and populates with AI analyses.
"""
import streamlit as st
import asyncio
from dotenv import load_dotenv

from src.orchestrator import MatchingOrchestrator
import traceback
import sys
import yaml

# Load env file
load_dotenv()

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
# Formatting
def to_yaml(obj) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return yaml.dump(
        obj,
        indent=4,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

def progress_update(message: str):
    status_placeholder.markdown(f"**{message}**")


# Async helper
def run_async(coroutine):
    """
    Safely run async code inside Streamlit.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coroutine)

# Traceability
def excepthook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)

sys.excepthook = excepthook

# Setup app controls
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None
if "report" not in st.session_state:
    st.session_state.report = None
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "summary_version" not in st.session_state:
    st.session_state.summary_version = 1

# ---------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Candidate–Job Analysis",
    layout="wide",
)

st.title("Candidate & Job Analysis")

st.markdown(
    """
    Provide candidate and job details.
    The system will analyse candidate fit for the role and provide a summary with recommendation.
    """
)

# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    candidate_id = st.text_input("Candidate unique identifier", placeholder="c_001")
    resume_source = st.text_input(
        "Resume source (URL or file path)",
        placeholder="https://example.com/resume.pdf"
    )

with col2:
    job_id = st.text_input("Job unique identifier", placeholder="j_001")
    job_source = st.text_input(
        "Job description source (URL or file path)",
        placeholder="https://example.com/job.html"
    )

analyse_btn = st.button("Run Analysis")

# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

if analyse_btn:
    if not resume_source or not job_source:
        st.error("Please provide both resume and job sources.")
    else:
        st.session_state.analysis_running = True
        st.session_state.orchestrator = MatchingOrchestrator()
        

        with st.spinner("Running analysis pipeline..."):
            try:
                status_placeholder = st.empty()
                report = run_async(
                    st.session_state.orchestrator.analyse(
                        resume_source=resume_source,
                        job_source=job_source,
                        candidate_id=candidate_id,
                        job_id=job_id,
                        progress_callback=progress_update
                    )
                )
                st.session_state.report = report
                st.session_state.current_summary = report.summary.summary

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        status_placeholder.empty()
        st.success("Analysis completed")

# -----------------------------------------------------------------
# Display parsed outputs
# -----------------------------------------------------------------
if st.session_state.report is not None:
    report = st.session_state.report
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Fit Analysis", "Parsed Resume", "Parsed Job"])

    with tab1:
        left, right = st.columns([3, 1])
        with left:
            st.subheader("Candidate Details")
            st.code(to_yaml(report.candidate_details), language="yaml")

            st.write(f'''**Summary Version** : {st.session_state.summary_version}''')
            st.write(st.session_state.current_summary)

        with right:
            # Section for recruiter comments    
            st.subheader("Recruiter Comments")

            recruiter_feedback = st.text_area(
                "Add comments to refine the summary",
                placeholder="E.g. Leadership exposure seems limited. Culture fit is strong."
            )

            regenerate = st.button("Re-generate Summary")
        if regenerate:
            if not recruiter_feedback.strip():
                st.warning("Please add comments before regenerating.")
            else:
                with st.spinner("Re-generating summary with recruiter input"):
                    updated_summary = run_async(
                        st.session_state.orchestrator.regenerate_summary(
                            report=st.session_state.report,
                            recruiter_feedback=recruiter_feedback
                        )
                    )
                st.session_state.report.summary = updated_summary
                st.session_state.current_summary = updated_summary.summary
                # st.session_state.report.summary.summary = updated_summary
                # st.session_state.current_summary = updated_summary
                st.session_state.summary_version += 1
            with left:
                    # Update with new values
                    st.write(f'''**Summary Version** : {st.session_state.summary_version}''')
                    st.write(st.session_state.current_summary)

                    st.success("Summary updated")

    with tab2:
        st.subheader("Fit Analysis")
        st.code(to_yaml(report.fit_analysis), language="yaml")

    with tab3:
        st.subheader("Parsed Resume")
        st.code(to_yaml(report.resume), language="yaml")

    with tab4:
        st.subheader("Parsed Job Description")
        st.code(to_yaml(report.job), language="yaml")
