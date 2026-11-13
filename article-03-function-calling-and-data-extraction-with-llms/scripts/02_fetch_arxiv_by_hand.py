# The two tools the Reader will use, written by hand
# pip install requests pypdf
#
# fetch_arxiv downloads a paper once and keeps its text in a local cache.
# get_section returns one section of it. If the name is wrong, the error text
# lists the real section names, so a model can correct itself.
import re
import sys
import time
from pathlib import Path

import requests
from pypdf import PdfReader

ARGS = sys.argv[1:] if "__file__" in globals() else []  # a notebook passes its own arguments, ignore them
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # a notebook has no __file__

CACHE = SCRIPT_DIR.parent / ".paper_cache"
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
# "1 Introduction", "4.2 Results", plus the unnumbered headings papers use
# Section numbers start at 1 ("0.42 Ounce" is body text) and headings have no brackets
HEADING = re.compile(r"^((?!0\.)\d{1,2}(\.\d{1,2})*\.?\s+[A-Z][^\n()]{2,70}|Abstract|References|Appendix[^\n]{0,40})$")

PAPERS = ["2310.13548", "2408.02442", "2501.10868", "2303.11366", "2303.17651",
          "2302.04761", "2210.03629", "2201.11903", "1706.03762", "2307.03172"]


def fetch_arxiv(arxiv_id: str) -> str:
    """Return the paper's text. Downloads on the first call, then reads the cache."""
    if not ARXIV_ID.match(arxiv_id):
        return f"Error: '{arxiv_id}' is not an arXiv ID like 2408.02442."
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{arxiv_id}.txt"
    if cached.exists():
        return cached.read_text()
    pdf = CACHE / f"{arxiv_id}.pdf"
    response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60,
                        headers={"User-Agent": "paper-lab/0.1 (personal learning project)"})
    if response.status_code != 200:
        return f"Error: arXiv answered {response.status_code} for {arxiv_id}."
    pdf.write_bytes(response.content)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
    cached.write_text(text)
    pdf.unlink()  # keep only the text
    time.sleep(3)  # arXiv asks automated clients to go slowly
    return text


def split_into_sections(text: str) -> dict:
    """Cut the text at lines that look like headings."""
    sections, name, lines = {}, "Start", []
    for line in text.splitlines():
        line = line.strip()
        if HEADING.match(line):
            sections[name] = "\n".join(lines).strip()
            name, lines = line, []
        else:
            lines.append(line)
    sections[name] = "\n".join(lines).strip()
    return {heading: body for heading, body in sections.items() if body}


def normalize_heading(heading: str) -> str:
    """Lowercase and drop everything but letters and digits: 'I NTRODUCTION' -> 'introduction'."""
    return re.sub(r"[^a-z0-9]", "", heading.lower())


def get_section(arxiv_id: str, name: str) -> str:
    """Return the first section whose heading contains `name`, ignoring case and spacing."""
    text = fetch_arxiv(arxiv_id)
    if text.startswith("Error:"):
        return text
    sections = split_into_sections(text)
    for heading, body in sections.items():
        if normalize_heading(name) in normalize_heading(heading):
            return body
    return f"Error: no section called '{name}'. Sections: {', '.join(sections)}"


if __name__ == "__main__":
    arxiv_ids = ARGS or PAPERS
    print(f"{'paper':<11} {'chars':>7}  {'sections':>8}")
    for arxiv_id in arxiv_ids:
        text = fetch_arxiv(arxiv_id)
        if text.startswith("Error:"):
            print(f"{arxiv_id:<11} {text}")
        else:
            print(f"{arxiv_id:<11} {len(text):>7}  {len(split_into_sections(text)):>8}")
