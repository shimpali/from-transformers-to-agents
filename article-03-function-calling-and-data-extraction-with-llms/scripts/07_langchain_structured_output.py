# The same claim extraction, three ways of getting a filled-in form back
# pip install requests pydantic langchain-ollama     (and run Ollama with granite4.2:3b)
#
#   hand-written                ask in the prompt, validate with Pydantic, retry with the
#                               error shown (script 03)
#   LangChain json_schema       with_structured_output(method="json_schema"): the schema goes
#                               to Ollama as `format`, so the model can only write text that fits
#   LangChain function_calling  with_structured_output(method="function_calling"): the schema is
#                               offered as a tool and the model "calls" it
#
# All three read the same abstracts with the same instructions, at temperature 0, thinking off.
# Needs the papers cached by 02_fetch_arxiv_by_hand.py.
# A method that returns nothing usable is counted as invalid and its reason is printed, never hidden.
#
#   python3 07_langchain_structured_output.py              all 10 papers
#   python3 07_langchain_structured_output.py 2201.11903   one paper
import hashlib
import importlib.util
import json
import sys
import textwrap
import time
from pathlib import Path

import requests

ARGS = sys.argv[1:] if "__file__" in globals() else []  # a notebook passes its own arguments, ignore them
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # a notebook has no __file__

sys.dont_write_bytecode = True  # do not leave a __pycache__ folder next to the scripts
HERE = SCRIPT_DIR.parent
MODEL = "granite4.2:3b"
MAX_OUT = 4000  # token cap per call

# Reuse the schema, prompt, abstract finder, parser and scorer from script 03
_reader_spec = importlib.util.spec_from_file_location("s03", (SCRIPT_DIR / "03_extract_claims.py"))
reader = importlib.util.module_from_spec(_reader_spec)
_reader_spec.loader.exec_module(reader)
SYSTEM, Claims, MAX_RETRIES = reader.SYSTEM, reader.Claims, reader.MAX_RETRIES

OPTIONS = {"temperature": 0, "seed": 1, "num_ctx": 8192, "num_predict": MAX_OUT}


def show_progress(text):
    print("    " + text, file=sys.stderr, flush=True)


def read_cache(key):
    cache_file = HERE / ".model_cache" / f"{key}.json"
    return json.loads(cache_file.read_text()) if cache_file.exists() else None


def write_cache(key, value):
    cache_folder = HERE / ".model_cache"
    cache_folder.mkdir(exist_ok=True)
    (cache_folder / f"{key}.json").write_text(json.dumps(value))


def make_cache_key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]


# ---- arm 1: hand-written ----
def ask_ollama(messages, call_label):
    payload = {"model": MODEL, "messages": messages, "stream": False, "think": False,
               "keep_alive": "10m", "options": OPTIONS}
    key = make_cache_key("07-hand", payload)
    cached = read_cache(key)
    if cached:
        show_progress(f"{call_label}: from cache")
        return cached
    show_progress(f"{call_label}: asking Ollama ...")
    started = time.time()
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
    if response.status_code != 200:  # loud, not cached
        show_progress(f"{call_label}: REFUSED by Ollama ({response.status_code}): {response.text[:200]}")
        return {"text": "", "tokens_out": 0, "secs": 0, "error": response.text[:200]}
    data = response.json()
    result = {"text": data["message"]["content"], "tokens_out": data.get("eval_count", 0),
           "secs": round(time.time() - started, 1), "error": ""}
    write_cache(key, result)
    show_progress(f"{call_label}: done in {result['secs']:.0f} s ({result['tokens_out']} tokens out)")
    return result


def run_hand_written_method(abstract, arxiv_id):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": abstract}]
    record = {"claims": None, "first_ok": False, "tries": 0, "tokens_out": 0, "secs": 0.0, "error": ""}
    for attempt in range(MAX_RETRIES + 1):
        result = ask_ollama(messages, f"{arxiv_id} hand-written try {attempt + 1}")
        record["tries"] = attempt + 1
        record["tokens_out"] += result["tokens_out"]
        record["secs"] += result["secs"]
        if result.get("error"):
            record["error"] = result["error"]
            return record
        claims, error = reader.parse_claims(result["text"])
        if claims:
            record["claims"], record["first_ok"] = claims, attempt == 0
            return record
        record["error"] = error
        messages += [{"role": "assistant", "content": result["text"]},
                     {"role": "user", "content": f"That was not valid: {error}. Reply with corrected JSON only."}]
    return record


# ---- arms 2 and 3: LangChain ----
def run_langchain_method(method, abstract, arxiv_id):
    from langchain_ollama import ChatOllama  # imported here so arm 1 works without LangChain

    call_label = f"{arxiv_id} lc {method}"
    # function_calling got a new cache version: its first run hid WHY a reply failed
    key = make_cache_key("07-lc2" if method == "function_calling" else "07-lc", method, SYSTEM, abstract, OPTIONS, MODEL)
    cached = read_cache(key)
    if cached:
        show_progress(f"{call_label}: from cache")
    else:
        show_progress(f"{call_label}: asking Ollama ...")
        llm = ChatOllama(model=MODEL, temperature=0, seed=1, num_ctx=8192, num_predict=MAX_OUT,
                         reasoning=False, keep_alive="10m")
        structured = llm.with_structured_output(Claims, method=method, include_raw=True)
        started = time.time()
        try:
            result = structured.invoke([("system", SYSTEM), ("human", abstract)])
            usage = getattr(result["raw"], "usage_metadata", None) or {}
            parsed = result["parsed"]
            error = " ".join(str(result["parsing_error"]).split())[:200] if result["parsing_error"] else ""
            if parsed is None and not error:
                # LangChain returns None and no error when the model never called the tool
                raw = result["raw"]
                error = (f"LangChain gave None and no error: the model made {len(raw.tool_calls or [])} tool call(s) "
                         f"and wrote: {' '.join((raw.content or '').split())[:150]!r}")
            cached = {"claims": parsed.model_dump() if parsed else None, "error": error,
                   "tokens_out": usage.get("output_tokens", 0), "secs": round(time.time() - started, 1)}
        except Exception as problem:  # for example Ollama refuses tools for this model: loud, not cached
            show_progress(f"{call_label}: FAILED before any reply: {type(problem).__name__}: {str(problem)[:160]}")
            return {"claims": None, "first_ok": False, "tries": 1, "tokens_out": 0, "secs": 0.0,
                    "error": f"{type(problem).__name__}: {str(problem)[:160]}"}
        write_cache(key, cached)
        show_progress(f"{call_label}: done in {cached['secs']:.0f} s ({cached['tokens_out']} tokens out)"
            + ("" if cached["claims"] else f", NO VALID RESULT: {cached['error'][:80]}"))
    claims = Claims.model_validate(cached["claims"]) if cached["claims"] else None
    return {"claims": claims, "first_ok": claims is not None, "tries": 1,
            "tokens_out": cached["tokens_out"], "secs": cached["secs"], "error": cached["error"]}


# The three methods compared. "LangChain" rows go through LangChain's with_structured_output.
METHODS = [("hand-written", run_hand_written_method),
           ("LangChain json_schema", lambda abstract, arxiv_id: run_langchain_method("json_schema", abstract, arxiv_id)),
           ("LangChain function_calling",
            lambda abstract, arxiv_id: run_langchain_method("function_calling", abstract, arxiv_id))]
# valid first try = the first reply passed validation; valid = it passed after any retries
HEAD = f"{'method':<26} {'valid first try':>15} {'valid':>5} {'found':>6} {'tokens out':>10} {'seconds':>7}"

if __name__ == "__main__":
    reference_claims = json.loads((HERE / "reference_claims.json").read_text())
    reference_claims = {arxiv_id: claims for arxiv_id, claims in reference_claims.items() if not arxiv_id.startswith("_")}
    papers = ARGS or list(reference_claims)
    rows = []
    for arxiv_id in papers:
        abstract = reader.get_abstract(arxiv_id)
        print(f"\npaper {arxiv_id}")
        print(HEAD)
        for name, run_method in METHODS:
            record = run_method(abstract, arxiv_id)
            found = sum(reader.score_claims(record["claims"], reference_claims[arxiv_id])) if record["claims"] else 0
            is_valid = record["claims"] is not None
            note = "" if is_valid and record["first_ok"] else ("needed retry" if is_valid else record["error"][:140])
            found_text = f"{found}/3"
            print(f"{name:<26} {'yes' if record['first_ok'] else 'NO':>15} {'yes' if is_valid else 'NO':>5} "
                  f"{found_text:>6} {record['tokens_out']:>10} {record['secs']:>7.0f}")
            if note:  # the reason goes on its own lines so the table stays narrow
                print(textwrap.fill(note, 76, initial_indent="    ", subsequent_indent="    "))
            rows.append({"paper": arxiv_id, "method": name, "valid_first": record["first_ok"], "valid": is_valid, "found": found,
                         "tokens_out": record["tokens_out"], "secs": record["secs"], "tries": record["tries"],
                         "error": record["error"]})
    print(f"\nall {len(papers)} papers")
    print(HEAD)
    for name, _ in METHODS:
        method_rows = [row for row in rows if row["method"] == name]
        paper_count = len(method_rows)
        first = f"{sum(row['valid_first'] for row in method_rows)}/{paper_count}"
        valid = f"{sum(row['valid'] for row in method_rows)}/{paper_count}"
        found_text = f"{sum(row['found'] for row in method_rows)}/{3 * paper_count}"
        print(f"{name:<26} {first:>15} {valid:>5} {found_text:>6} "
              f"{sum(row['tokens_out'] for row in method_rows):>10} {sum(row['secs'] for row in method_rows):>7.0f}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "07_langchain_structured_output.json").write_text(
        json.dumps({"model": MODEL, "think": False, "rows": rows}, indent=1))
