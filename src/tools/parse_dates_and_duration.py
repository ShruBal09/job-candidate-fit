"""
Tool definition for Dates parsing and duration calculation
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime
from dateutil import parser

def parse_dates_and_duration(
    start_date: Optional[str],
    end_date: Optional[str]
) -> Dict[str, Any]:
    """
    Parse start and end dates from resume-style strings and compute duration in years.

    Special handling:
    - If parsing fails or dates are missing → duration is None
    - "present", "current", "now" → datetime.now()
    Args:

    Returns:
        {
            "start_date_parsed": ISO string or None,
            "end_date_parsed": ISO string or None,
            "duration_years": float or None,
            "note": str
        }

    """

    def _parse(d: Optional[str]) -> Optional[datetime]:
        if not d:
            return None
        d_clean = d.strip().lower()
        if d_clean in {"present", "current", "now"}:
            return datetime.now()
        try:
            return parser.parse(d, default=datetime(1900, 1, 1))
        except Exception:
            return None

    start = _parse(start_date)
    end = _parse(end_date)

    if not start or not end or end < start:
        return {
            "start_date_parsed": start.isoformat() if start else None,
            "end_date_parsed": end.isoformat() if end else None,
            "duration_years": None,
            "note": "Could not reliably compute duration from provided dates."
        }

    duration_years = round((end - start).days / 365, 2)

    return {
        "start_date_parsed": start.isoformat(),
        "end_date_parsed": end.isoformat(),
        "duration_years": duration_years,
        "note": "Duration computed from parsed start/end dates."
    }
