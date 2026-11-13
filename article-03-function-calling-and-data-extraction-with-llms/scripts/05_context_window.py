# Context window: the whole paper at once, or one section at a time
# pip install requests pydantic     (and run Ollama with granite4.2:3b)
#
# Three ways to read the same paper:
#   small window  whole paper, Ollama's usual 4096-token window (Ollama refuses it, loudly)
#   big window    whole paper, a 20480-token window
#   sections      one call per section, claims merged afterwards
# Section calls may write at most MAX_OUT tokens (thinking included). A call that hits
# the cap is counted and shown in the table, never hidden.
# Needs the papers cached by 02_fetch_arxiv_by_hand.py.
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
PAPERS = ["1706.03762", "2303.11366"]
SMALL, BIG = 4096, 20480
MAX_RETRIES = 2  # per chunk, then the chunk counts as invalid
MAX_OUT = 4000  # cap for the section calls
HEADING = re.compile(r"^((?!0\.)\d{1,2}(\.\d{1,2})*\.?\s+[A-Z][^\n()]{2,70}|Abstract|References|Appendix[^\n]{0,40})$")

SYSTEM = ("You extract testable claims from paper text. Reply with JSON only, shaped like "
          '{"claims": [{"claim": "...", "how_to_test": "...", "expected_result": "..."}]}. '
          "Give at most 5 claims, the most important first. Put exact numbers from the text into "
          "claim and expected_result.")


class Claim(BaseModel):
    claim: str
    how_to_test: str
    expected_result: str


class Claims(BaseModel):
    claims: list[Claim]


def show_progress(text):
    print("    " + text, file=sys.stderr, flush=True)


def ask_ollama(messages, num_ctx, max_out=None, call_label=""):
    """One call to a local Ollama model at temperature 0, cached on disk."""
    payload = {"model": MODEL, "messages": messages, "stream": False, "keep_alive": "10m",
               "options": {"temperature": 0, "seed": 1, "num_ctx": num_ctx}}
    if max_out:
        payload["options"]["num_predict"] = max_out  # stop a runaway call
    key = hashlib.sha256(("v2" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()[:20]
    cache = HERE / ".model_cache" / f"{key}.json"
    if cache.exists():
        result = json.loads(cache.read_text())
        show_progress(f"{call_label}: from cache ({result['tokens_in']} tokens in, {result['tokens_out']} tokens out, {result['secs']} s when it ran)")
        return result
    show_progress(f"{call_label}: asking Ollama (window {num_ctx}, cap {max_out or 'none'}) ...")
    started = time.time()
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
    except requests.exceptions.Timeout:  # a stuck call must not freeze the whole run
        show_progress(f"{call_label}: TIMED OUT after 600 s")
        return {"text": "", "error": "no answer after 600 s", "thinking_chars": 0,
                "tokens_in": 0, "tokens_out": 0, "secs": 600, "capped": False}
    if response.status_code == 400:  # Ollama refused the request: keep its reason as a result
        show_progress(f"{call_label}: REFUSED by Ollama after {time.time() - started:.0f} s")
        result = {"text": "", "error": response.text[:300], "thinking_chars": 0, "tokens_in": 0, "tokens_out": 0, "secs": 0, "capped": False}
        return result  # not cached, so a fixed setup is retried next time
    if response.status_code != 200:  # loud: show Ollama's own reason, not just the status code
        raise RuntimeError(f"Ollama answered {response.status_code}: {response.text[:300]}")
    data = response.json()
    result = {"text": data["message"]["content"], "thinking_chars": len(data["message"].get("thinking") or ""),
           "tokens_in": data.get("prompt_eval_count", 0), "tokens_out": data.get("eval_count", 0),
           "secs": round(time.time() - started, 1), "capped": data.get("done_reason") == "length"}
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(result))
    show_progress(f"{call_label}: done in {result['secs']:.0f} s ({result['tokens_in']} tokens in, {result['tokens_out']} tokens out, "
        f"thinking {result['thinking_chars']} characters" + (", HIT THE CAP" if result["capped"] else "") + ")")
    return result


def load_paper_text(arxiv_id):
    raw = (HERE / ".paper_cache" / f"{arxiv_id}.txt").read_text()
    return unicodedata.normalize("NFKC", raw)


def split_paper_into_chunks(text, min_chars=1500, max_chars=12000):
    """Cut at headings, glue short sections to the next one, drop references."""
    parts, name, lines = [], "start", []
    for line in text.splitlines():
        if HEADING.match(line.strip()):
            parts.append((name, "\n".join(lines)))
            name, lines = line.strip(), []
        else:
            lines.append(line)
    parts.append((name, "\n".join(lines)))
    chunks, buffer = [], ""
    for name, body in parts:
        if name.lower().startswith(("references", "appendix")):
            break
        buffer += f"\n{name}\n{body}"
        if len(buffer) >= min_chars:
            chunks.append(buffer[:max_chars])
            buffer = ""
    if buffer.strip():
        chunks.append(buffer[:max_chars])
    return [re.sub(r"[ \t]+", " ", chunk).strip() for chunk in chunks]


def parse_claims(reply):
    """Return (claims, error). Models often wrap JSON in a code fence, so strip that."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    try:
        return Claims.model_validate_json(cleaned).claims, None
    except (ValidationError, ValueError) as problem:
        return None, " ".join(str(problem).split())[:200]


def extract_claims_from_chunk(chunk, num_ctx, max_out, call_label=""):
    """One chunk: ask, validate, retry with the error shown (capped). A reply cut off by the
    token cap is not retried, because asking again would run into the same cap."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": chunk}]
    totals = {"claims": [], "refused": "", "invalid": False, "capped": False, "retries": 0, "first_in": 0,
         "tokens_in": 0, "tokens_out": 0, "think": 0, "secs": 0}
    for attempt in range(MAX_RETRIES + 1):
        result = ask_ollama(messages, num_ctx, max_out, f"{call_label} try {attempt + 1}")
        if attempt == 0:
            totals["first_in"] = result["tokens_in"]
        totals["tokens_in"] += result["tokens_in"]; totals["tokens_out"] += result["tokens_out"]
        totals["think"] += result.get("thinking_chars", 0); totals["secs"] += result["secs"]
        if result.get("error"):
            totals["refused"] = result["error"]
            return totals
        claims, error = parse_claims(result["text"])
        if claims is not None:
            totals["claims"] = claims
            return totals
        totals["capped"] = totals["capped"] or bool(result.get("capped"))
        if result.get("capped") or attempt == MAX_RETRIES:
            break
        messages += [{"role": "assistant", "content": result["text"]},
                     {"role": "user", "content": f"That was not valid: {error}. Reply with corrected JSON only."}]
        totals["retries"] += 1
    totals["invalid"] = True  # nothing usable: counted and shown, never invented
    return totals


def run_reading_method(label, chunks, num_ctx, references, max_out=None):
    results = [extract_claims_from_chunk(chunk, num_ctx, max_out,
                   f"{label} {number}/{len(chunks)} ({len(chunk)} chars, '{' '.join(chunk.split())[:28]}...')")
               for number, chunk in enumerate(chunks, 1)]
    claims = [claim for result in results for claim in result["claims"]]
    claim_texts = [claim.claim + " " + claim.expected_result for claim in claims]
    found = sum(any(all(re.search(pattern, text, re.I) for pattern in reference["facts"]) for text in claim_texts)
                for reference in references)
    tokens_sent = sum(len(chunk) for chunk in chunks) / 4  # rough guess: 4 characters per token
    tokens_read = sum(result["first_in"] for result in results)
    errors = [result["refused"] for result in results if result["refused"]]
    return {"how": label, "calls": len(results), "error": errors[0] if errors else "", "found": found,
            "claims": len(claims),
            "tokens_in": sum(result["tokens_in"] for result in results),
            "tokens_out": sum(result["tokens_out"] for result in results),
            "secs": round(sum(result["secs"] for result in results)),
            "capped": sum(result["capped"] for result in results),
            "invalid": sum(result["invalid"] for result in results),
            "retries": sum(result["retries"] for result in results),
            "think_chars": sum(result["think"] for result in results),
            "cut": tokens_read < 0.8 * tokens_sent}  # Ollama read clearly fewer tokens than we sent


if __name__ == "__main__":
    reference_claims = json.loads((HERE / "reference_claims.json").read_text())
    rows = []
    # capped = calls stopped by the token cap, result of all calls made (4/13 = 4 of 13 calls),
    # bad = replies that failed validation,
    # think = characters of hidden reasoning
    for arxiv_id in ARGS or PAPERS:
        text = load_paper_text(arxiv_id)
        print(f"\npaper {arxiv_id}")
        print(f"{'how':<8} {'found':>5} {'claims':>6} {'tokens in':>9} {'tokens out':>10} {'seconds':>7} "
              f"{'capped':>6} {'bad':>5} {'retry':>5} {'think':>6}")
        for label, chunks, window_size in (("small", [text], SMALL), ("big", [text], BIG),
                                   ("sections", split_paper_into_chunks(text), 8192)):
            row = run_reading_method(label, chunks, window_size, reference_claims[arxiv_id], MAX_OUT if label == "sections" else None)
            rows.append({"paper": arxiv_id, **row})
            capped, bad = f"{row['capped']}/{row['calls']}", f"{row['invalid']}/{row['calls']}"
            print(f"{label:<8} {row['found']:>3}/3 {row['claims']:>6} {row['tokens_in']:>9} "
                  f"{row['tokens_out']:>10} {row['secs']:>7} {capped:>6} {bad:>5} {row['retries']:>5} {row['think_chars']:>6}")
            if row["error"]:  # keep Ollama's own reason, shortened to its message
                reason = re.search(r"request \(\d+ tokens\) exceeds[^\\\"]*", row["error"])
                print("  Ollama refused the call:")
                print("  " + (reason.group(0).split(", try")[0] if reason else row["error"][:70]))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "05_context_window.json").write_text(json.dumps({"model": MODEL, "rows": rows}, indent=1))
