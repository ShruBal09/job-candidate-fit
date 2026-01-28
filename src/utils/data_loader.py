"""
Read PDF, HTML and URLs into text
"""
import pymupdf 
import requests
from bs4 import BeautifulSoup
from pathlib import Path

def load_pdf(path: str) -> str:
    """
    Load and extract text from a PDF file using PyMuPDF.
    """
    text = ""
    with pymupdf.open(path) as doc:
        for page in doc:
            text += page.get_text().replace("\u200b", "")
    return text

def load_html(path: str) -> str:
    """
    Load and extract text from a local HTML file.
    """

    html = Path(path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")

def load_url(url: str) -> str:
    """
    Fetch HTML from a URL and extract visible text.
    """
    resp = requests.get(url, headers={"User-Agent": "python-etl/1.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator=" ")

def load_text_file(path: str) -> str:
    """
    Read a text file.
    """
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def load_document(source: str) -> str:
    """
    Smart loader:
      - URL => fetch
      - .pdf => pdf extractor
      - .html/.htm => html extractor
      - else => plain text
    """
    s = source.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return load_url(s)

    p = Path(s)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {s}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(str(p))
    if ext in {".html", ".htm"}:
        return load_html(str(p))
    return load_text_file(str(p))
