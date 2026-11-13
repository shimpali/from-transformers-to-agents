# The Reader: claims from a paper as validated JSON
# pip install requests pydantic     (and run Ollama: ollama pull granite4.2:3b)
#
# The model gets the abstract and a description of the JSON shape. Python checks
# the reply against a schema, retries with the error shown (capped), and then
# scores the claims against a hand-written reference list.
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests
from pydantic import BaseModel, ValidationError

ARGS = sys.argv[1:] if "__file__" in globals() else []  # a notebook passes its own arguments, ignore them
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # a notebook has no __file__

HERE = SCRIPT_DIR.parent
MODEL = "granite4.2:3b"
MAX_RETRIES = 2  # the cap: first try plus two retries, then the paper counts as failed

SYSTEM = ("You extract testable claims from a paper. Reply with JSON only, shaped like "
          '{"claims": [{"claim": "...", "how_to_test": "...", "expected_result": "..."}]}. '
          "Put exact numbers from the text into claim and expected_result.")


class Claim(BaseModel):
    claim: str
    how_to_test: str
    expected_result: str


class Claims(BaseModel):
    claims: list[Claim]


def ask_ollama(messages, temperature=0.0, seed=0, num_ctx=8192, model=None):
    """One call to a local Ollama model. Every answer is cached on disk."""
    payload = {"model": model or MODEL, "messages": messages, "stream": False, "keep_alive": "10m",
               "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}}
    key = hashlib.sha256(("v2" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()[:20]
    cache = HERE / ".model_cache" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    started = time.time()
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=900)
    if response.status_code != 200:  # loud: show Ollama's own reason, not just the status code
        raise RuntimeError(f"Ollama answered {response.status_code}: {response.text[:300]}")
    data = response.json()
    result = {"text": data["message"]["content"], "thinking_chars": len(data["message"].get("thinking") or ""),
           "tokens_in": data.get("prompt_eval_count", 0),
           "tokens_out": data.get("eval_count", 0), "secs": round(time.time() - started, 1)}
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


# In these PDFs a figure or footnote sits between the abstract and the introduction
ABSTRACT_ENDS = {"2201.11903": "A: The cafeteria", "1706.03762": "∗Equal contribution"}


def get_abstract(arxiv_id):
    """Abstract text from the paper cached by 02_fetch_arxiv_by_hand.py."""
    # NFKC turns PDF ligatures such as the single "fi" glyph back into plain letters
    text = unicodedata.normalize("NFKC", (HERE / ".paper_cache" / f"{arxiv_id}.txt").read_text())
    match = re.search(r"abstract", text, re.I)
    rest = text[match.end():] if match else text
    end = re.search(r"\n\s*1\.?\s+(I\s?NTRODUCTION|Introduction)", rest)
    stop = end.start() if end else 3000
    marker = rest.find(ABSTRACT_ENDS.get(arxiv_id, "\0"))
    if marker != -1:
        stop = min(stop, marker)
    return re.sub(r"\s+", " ", rest[:stop])[:3000]


def parse_claims(reply):
    """Return (claims, error). Models often wrap JSON in a code fence, so strip that."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    try:
        return Claims.model_validate_json(cleaned), None
    except (ValidationError, ValueError) as problem:
        return None, str(problem).splitlines()[0][:160]


def extract_claims(text, temperature=0.0, seed=0, num_ctx=8192):
    """Ask, validate, retry with the error shown, give up after MAX_RETRIES."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    stats = {"tokens_in": 0, "tokens_out": 0, "secs": 0.0, "tries": 0}
    for attempt in range(MAX_RETRIES + 1):
        result = ask_ollama(messages, temperature, seed, num_ctx)
        for field in ("tokens_in", "tokens_out", "secs"):
            stats[field] += result[field]
        stats["tries"] = attempt + 1
        claims, error = parse_claims(result["text"])
        if claims:
            return claims, stats
        messages += [{"role": "assistant", "content": result["text"]},
                     {"role": "user", "content": f"That was not valid: {error}. Reply with corrected JSON only."}]
    return None, stats  # failed: never invent a result


def score_claims(claims, references):
    """How many reference claims have one extracted claim matching every pattern."""
    claim_texts = [(claim.claim + " " + claim.expected_result) for claim in claims.claims] if claims else []
    return [any(all(re.search(pattern, text, re.I) for pattern in reference["facts"]) for text in claim_texts)
            for reference in references]


if __name__ == "__main__":
    reference_claims = json.loads((HERE / "reference_claims.json").read_text())
    reference_claims = {arxiv_id: claims for arxiv_id, claims in reference_claims.items() if not arxiv_id.startswith("_")}
    model = ARGS[0] if ARGS else MODEL
    MODEL = model
    rows, found, total = [], 0, 0
    print(f"{'paper':<11} {'tries':>5} {'valid':>5} {'found':>5} {'tokens in':>9} {'tokens out':>10} {'seconds':>7}")
    for arxiv_id, references in reference_claims.items():
        claims, stats = extract_claims(get_abstract(arxiv_id))
        hits = score_claims(claims, references)
        found, total = found + sum(hits), total + len(hits)
        print(f"{arxiv_id:<11} {stats['tries']:>5} {'yes' if claims else 'NO':>5} {sum(hits):>3}/{len(hits)} "
              f"{stats['tokens_in']:>9} {stats['tokens_out']:>10} {stats['secs']:>7.0f}")
        rows.append({"paper": arxiv_id, "valid": bool(claims), "hits": hits, **stats})
    print(f"\nclaims found: {found}/{total}  valid JSON: {sum(row['valid'] for row in rows)}/{len(rows)}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "03_extract.json").write_text(json.dumps({"model": model, "rows": rows}, indent=1))
