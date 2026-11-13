# Temperature: the same extraction five times at 0 and at 1
# pip install requests pydantic     (and run Ollama with granite4.2:3b)
#
# Each run uses a different seed. At temperature 0 the seed should not matter,
# so the five answers should match. At temperature 1 every run is a new spin.
# Needs the abstracts cached by 02_fetch_arxiv_by_hand.py.
#
#   python3 04_temperature.py            reads the saved same-seed counts (fast)
#   python3 04_temperature.py --fresh    makes the 10 same-seed calls again (slow, may differ)
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import requests
from pydantic import BaseModel, ValidationError

ARGS = sys.argv[1:] if "__file__" in globals() else []  # a notebook passes its own arguments, ignore them
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # a notebook has no __file__

HERE = SCRIPT_DIR.parent
MODEL = "granite4.2:3b"
MAX_RETRIES = 2
PAPERS = ["1706.03762", "2210.03629", "2303.11366"]  # all found, partly found, mostly missed in script 03
SEEDS = [1, 2, 3, 4, 5]
ABSTRACT_ENDS = {"1706.03762": "∗Equal contribution"}

SYSTEM = ("You extract testable claims from a paper. Reply with JSON only, shaped like "
          '{"claims": [{"claim": "...", "how_to_test": "...", "expected_result": "..."}]}. '
          "Put exact numbers from the text into claim and expected_result.")


class Claim(BaseModel):
    claim: str
    how_to_test: str
    expected_result: str


class Claims(BaseModel):
    claims: list[Claim]


def ask_ollama(messages, temperature, seed, use_cache=True):
    """One call to a local Ollama model, cached on disk by its full request."""
    payload = {"model": MODEL, "messages": messages, "stream": False, "keep_alive": "10m",
               "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192}}
    key = hashlib.sha256(("v2" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()[:20]
    cache = HERE / ".model_cache" / f"{key}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    started = time.time()
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=900)
    if response.status_code != 200:  # loud: show Ollama's own reason, not just the status code
        raise RuntimeError(f"Ollama answered {response.status_code}: {response.text[:300]}")
    data = response.json()
    result = {"text": data["message"]["content"], "thinking_chars": len(data["message"].get("thinking") or ""),
           "tokens_in": data.get("prompt_eval_count", 0), "tokens_out": data.get("eval_count", 0),
           "secs": round(time.time() - started, 1)}
    if use_cache:
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(result))
    return result


def get_abstract(arxiv_id):
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
    cleaned = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    try:
        return Claims.model_validate_json(cleaned), None
    except (ValidationError, ValueError) as problem:
        return None, str(problem).splitlines()[0][:160]


def extract_claims(text, temperature, seed):
    """Ask once; if the JSON is bad, retry with the error shown, up to MAX_RETRIES."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    first_reply = ask_ollama(messages, temperature, seed)
    result, tokens, tries = first_reply, first_reply["tokens_out"], 1
    for _ in range(MAX_RETRIES + 1):
        claims, error = parse_claims(result["text"])
        if claims or tries > MAX_RETRIES:
            return claims, first_reply, tokens, tries
        messages += [{"role": "assistant", "content": result["text"]},
                     {"role": "user", "content": f"That was not valid: {error}. Reply with corrected JSON only."}]
        result = ask_ollama(messages, temperature, seed)
        tokens, tries = tokens + result["tokens_out"], tries + 1
    return None, first_reply, tokens, tries


def matched_reference_positions(claims, references):
    """Which reference claims (by position) one extracted claim matches."""
    claim_texts = [(claim.claim + " " + claim.expected_result) for claim in claims.claims] if claims else []
    return tuple(position for position, reference in enumerate(references)
                 if any(all(re.search(pattern, text, re.I) for pattern in reference["facts"]) for text in claim_texts))


if __name__ == "__main__":
    reference_claims = json.loads((HERE / "reference_claims.json").read_text())
    FRESH = "--fresh" in ARGS  # repeat the slow same-seed check instead of reading its saved counts
    papers = [argument for argument in ARGS if not argument.startswith("--")] or PAPERS
    rows = []
    print(f"{'paper':<11} {'temp':>4} {'valid':>5} {'found':>5} {'agree':>5} {'distinct':>8} {'tokens out':>10}")
    for arxiv_id in papers:
        for temperature in (0.0, 1.0):
            runs = [extract_claims(get_abstract(arxiv_id), temperature, seed) for seed in SEEDS]
            found_sets = [matched_reference_positions(claims, reference_claims[arxiv_id]) for claims, _, _, _ in runs]
            first_try = sum(tries == 1 and claims is not None for claims, _, _, tries in runs)
            agree = Counter(found_sets).most_common(1)[0][1]  # runs that match the most common result
            distinct = len({first_reply["text"] for _, first_reply, _, _ in runs})  # how many different first replies
            mean_found = sum(len(found) for found in found_sets) / len(found_sets)
            mean_tokens = sum(tokens for _, _, tokens, _ in runs) / len(runs)
            print(f"{arxiv_id:<11} {temperature:>4.0f} {first_try:>3}/5 {mean_found:>5.1f} {agree:>3}/5 {distinct:>6}/5 {mean_tokens:>10.0f}")
            rows.append({"paper": arxiv_id, "temp": temperature, "valid_first_try": first_try, "mean_found": mean_found,
                         "agree": agree, "distinct": distinct, "mean_tokens_out": mean_tokens,
                         "sets": [list(found) for found in found_sets]})
    # Does temperature 0 repeat itself? Same prompt, same seed, five fresh calls, cache off.
    # Slow (10 calls) and not repeatable by design, so a rerun reads the counts saved by the
    # last fresh run. Add --fresh to run the calls again.
    print("\nsame seed, five fresh calls at temperature 0")
    saved = HERE / "results" / "04_temperature.json"
    same = json.loads(saved.read_text()).get("same_seed_distinct") if saved.exists() and not FRESH else None
    if same:
        for arxiv_id, distinct_replies in same.items():
            print(f"{arxiv_id:<11} distinct replies: {distinct_replies}/5   (saved from the last fresh run)")
    else:
        same = {}
        for arxiv_id in ("2210.03629", "2303.11366"):
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": get_abstract(arxiv_id)}]
            texts = [ask_ollama(messages, 0.0, 1, use_cache=False)["text"] for _ in range(5)]
            same[arxiv_id] = len(set(texts))
            print(f"{arxiv_id:<11} distinct replies: {same[arxiv_id]}/5")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "04_temperature.json").write_text(json.dumps({"model": MODEL, "rows": rows, "same_seed_distinct": same}, indent=1))
