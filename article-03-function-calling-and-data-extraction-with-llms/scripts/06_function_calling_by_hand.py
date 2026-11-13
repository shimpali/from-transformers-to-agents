# Function calling by hand: the model asks, our code runs the tool
# pip install requests pydantic pypdf     (and run Ollama with granite4.2:3b)
#
# The model never runs anything. It writes a request ("call get_section with these
# arguments"), THIS script runs the Python function and sends the result back, and
# the loop repeats until the model answers in plain text or the step cap is reached.
#
# Task per paper: read the abstract with the tools, then answer with claims as JSON.
# Needs the papers cached by 02_fetch_arxiv_by_hand.py.
#
#   python3 06_function_calling_by_hand.py                 all 10 papers
#   python3 06_function_calling_by_hand.py 2201.11903      one paper
#   python3 06_function_calling_by_hand.py --think off     ask Ollama to skip hidden reasoning
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import requests
from pydantic import BaseModel, ValidationError

ARGS = sys.argv[1:] if "__file__" in globals() else []  # a notebook passes its own arguments, ignore them
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # a notebook has no __file__

HERE = SCRIPT_DIR.parent
MODEL = "granite4.2:3b"
MAX_STEPS = 8        # model calls per paper, tool rounds and retries together
MAX_RETRIES = 2      # bad final JSON: ask again with the error shown, this many times
MAX_OUT = 6000       # token cap for one call, hidden reasoning included (3000 cut off a good answer)
SECTION_CHARS = 6000  # a tool result stays short so it cannot flood the context window
BAR = 8              # checkpoint: this many of 10 papers must succeed to stay with a 3B model

# Reuse the two hand-written tools from script 02 (its file name starts with a digit,
# so it cannot be imported the usual way)
sys.dont_write_bytecode = True  # do not leave a __pycache__ folder next to the scripts
_tools_spec = importlib.util.spec_from_file_location("tools02", (SCRIPT_DIR / "02_fetch_arxiv_by_hand.py"))
paper_tools = importlib.util.module_from_spec(_tools_spec)
_tools_spec.loader.exec_module(paper_tools)

SYSTEM = ("You are a research assistant with two tools. To read a paper: call fetch_arxiv_sections "
          "to see its sections, then call get_section to read one. Once you have read the abstract, "
          "stop calling tools and reply with JSON only, shaped like "
          '{"claims": [{"claim": "...", "how_to_test": "...", "expected_result": "..."}]}. '
          "Put exact numbers from the text into claim and expected_result.")

# What the model sees. Name, description and parameters are written into its prompt.
TOOLS = [
    {"type": "function", "function": {
        "name": "fetch_arxiv_sections",
        "description": "Download an arXiv paper and list its section names. Use this first, before reading a section.",
        "parameters": {"type": "object", "properties": {
            "arxiv_id": {"type": "string", "description": "arXiv ID such as 2408.02442"}},
            "required": ["arxiv_id"]}}},
    {"type": "function", "function": {
        "name": "get_section",
        "description": "Return the text of one section of an arXiv paper, for example Abstract or Introduction.",
        "parameters": {"type": "object", "properties": {
            "arxiv_id": {"type": "string", "description": "arXiv ID such as 2408.02442"},
            "name": {"type": "string", "description": "section name, for example Abstract"}},
            "required": ["arxiv_id", "name"]}}},
]


# ---- the tools themselves: ordinary Python, run by this script, never by the model ----
def fetch_arxiv_sections(arxiv_id: str) -> str:
    text = paper_tools.fetch_arxiv(arxiv_id)
    if text.startswith("Error:"):
        return text
    names = list(paper_tools.split_into_sections(text))
    return f"Paper {arxiv_id} fetched ({len(text)} characters). Sections: " + "; ".join(names[:30])


def get_section(arxiv_id: str, name: str) -> str:
    text = paper_tools.get_section(arxiv_id, name)
    return text[:SECTION_CHARS]


REGISTRY = {"fetch_arxiv_sections": fetch_arxiv_sections, "get_section": get_section}


def run_tool(name, args):
    """Run one requested call. Every failure comes back as text the model can read."""
    if name not in REGISTRY:
        return f"Error: there is no tool called '{name}'. Tools: {', '.join(REGISTRY)}.", False
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return "Error: the arguments were not valid JSON.", False
    try:
        return REGISTRY[name](**args), True
    except TypeError as problem:  # wrong or missing argument names
        return f"Error: bad arguments for {name}: {problem}", False


class Claim(BaseModel):
    claim: str
    how_to_test: str
    expected_result: str


class Claims(BaseModel):
    claims: list[Claim]


def parse_claims(reply):
    cleaned = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    try:
        return Claims.model_validate_json(cleaned).claims, None
    except (ValidationError, ValueError) as problem:
        return None, " ".join(str(problem).split())[:200]


def show_progress(text):
    print("    " + text, file=sys.stderr, flush=True)


THINK = None  # set by --think off


def ask_ollama_with_tools(messages, call_label):
    """One call to Ollama with the tool definitions attached. Cached on disk."""
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "stream": False, "keep_alive": "10m",
               "options": {"temperature": 0, "seed": 1, "num_ctx": 8192, "num_predict": MAX_OUT}}
    if THINK is not None:
        payload["think"] = THINK
    key = hashlib.sha256(("v1" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()[:20]
    cache = HERE / ".model_cache" / f"{key}.json"
    if cache.exists():
        result = json.loads(cache.read_text())
        show_progress(f"{call_label}: from cache ({result['secs']} s when it ran)")
        return result
    show_progress(f"{call_label}: asking Ollama ...")
    started = time.time()
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
    except requests.exceptions.Timeout:
        show_progress(f"{call_label}: TIMED OUT after 600 s")
        return {"error": "no answer after 600 s", "message": {}, "tokens_in": 0, "tokens_out": 0, "secs": 600, "capped": False}
    if response.status_code != 200:  # loud: keep Ollama's reason, do not cache
        show_progress(f"{call_label}: REFUSED by Ollama ({response.status_code}): {response.text[:200]}")
        return {"error": response.text[:300], "message": {}, "tokens_in": 0, "tokens_out": 0, "secs": 0, "capped": False}
    data = response.json()
    message = data["message"]
    result = {"message": {"content": message.get("content", ""), "tool_calls": message.get("tool_calls") or [],
                       "thinking_chars": len(message.get("thinking") or "")},
           "tokens_in": data.get("prompt_eval_count", 0), "tokens_out": data.get("eval_count", 0),
           "secs": round(time.time() - started, 1), "capped": data.get("done_reason") == "length"}
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(result))
    kind = "tool request" if result["message"]["tool_calls"] else "text answer"
    show_progress(f"{call_label}: done in {result['secs']:.0f} s, {kind} ({result['tokens_in']} tokens in, {result['tokens_out']} tokens out, "
        f"thinking {result['message']['thinking_chars']} characters" + (", HIT THE CAP" if result["capped"] else "") + ")")
    return result


def run_tool_loop(arxiv_id):
    """The loop. Returns a record of everything that happened."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Read the abstract of arXiv paper {arxiv_id} with your tools, "
                                            "then reply with its testable claims as JSON."}]
    record = {"paper": arxiv_id, "calls": [], "tool_errors": 0, "retries": 0, "model_calls": 0, "tokens_in": 0,
              "tokens_out": 0, "secs": 0.0, "claims": [], "ended": "", "read_ok": False}
    for step in range(1, MAX_STEPS + 1):
        reply = ask_ollama_with_tools(messages, f"{arxiv_id} step {step}/{MAX_STEPS}")
        record["model_calls"] += 1
        record["tokens_in"] += reply["tokens_in"]; record["tokens_out"] += reply["tokens_out"]; record["secs"] += reply["secs"]
        if reply.get("error"):
            record["ended"] = "ollama error: " + reply["error"][:100]
            return record
        message = reply["message"]
        if message["tool_calls"]:
            messages.append({"role": "assistant", "content": message["content"], "tool_calls": message["tool_calls"]})
            for call in message["tool_calls"]:
                name, args = call["function"]["name"], call["function"].get("arguments", {})
                tool_result, ran_ok = run_tool(name, args)
                if name == "get_section" and ran_ok and not tool_result.startswith("Error:"):
                    record["read_ok"] = True
                if not ran_ok or tool_result.startswith("Error:"):
                    record["tool_errors"] += 1
                record["calls"].append({"tool": name, "args": args, "result_start": tool_result[:120]})
                show_progress(f"  -> code ran {name}({json.dumps(args)[:90]}) -> {tool_result[:70]!r}")
                messages.append({"role": "tool", "tool_name": name, "content": tool_result})
            continue
        claims, error = parse_claims(message["content"])
        if claims is not None:
            record["claims"], record["ended"] = claims, "answered"
            return record
        if reply["capped"]:
            record["ended"] = "answer cut off by the token cap"
            return record
        if record["retries"] >= MAX_RETRIES:
            record["ended"] = "invalid JSON after retries"
            return record
        record["retries"] += 1
        messages += [{"role": "assistant", "content": message["content"]},
                     {"role": "user", "content": f"That was not valid: {error}. Reply with corrected JSON only."}]
    record["ended"] = f"step cap ({MAX_STEPS}) reached"
    return record


def count_found_claims(claims, references):
    claim_texts = [claim.claim + " " + claim.expected_result for claim in claims]
    return sum(any(all(re.search(pattern, text, re.I) for pattern in reference["facts"]) for text in claim_texts)
               for reference in references)


if __name__ == "__main__":
    args = ARGS
    if "--think" in args:
        position = args.index("--think")
        THINK = args[position + 1] != "off"
        del args[position:position + 2]
    reference_claims = json.loads((HERE / "reference_claims.json").read_text())
    reference_claims = {arxiv_id: claims for arxiv_id, claims in reference_claims.items() if not arxiv_id.startswith("_")}
    papers = args or list(reference_claims)
    print(f"{'paper':<11} {'tool calls':<10} {'errs':>4} {'retry':>5} {'calls':>5} {'found':>5} {'tokens out':>10} {'seconds':>7}  ended")
    rows, successes = [], 0
    for arxiv_id in papers:
        record = run_tool_loop(arxiv_id)
        record["found"] = count_found_claims(record["claims"], reference_claims[arxiv_id])
        record["success"] = record["read_ok"] and record["ended"] == "answered"
        successes += record["success"]
        call_sequence = ">".join(call["tool"].replace("fetch_arxiv_sections", "fetch").replace("get_section", "get")
                                 for call in record["calls"]) or "none"
        status = "OK" if record["success"] else "FAIL"
        print(f"{arxiv_id:<11} {call_sequence[:10]:<10} {record['tool_errors']:>4} {record['retries']:>5} {record['model_calls']:>5} "
              f"{record['found']:>3}/3 {record['tokens_out']:>10} {record['secs']:>7.0f}  {status}")
        if not record["success"]:
            print(f"            {record['ended']}")
        rows.append({**record, "claims": [claim.model_dump() for claim in record["claims"]]})
    print(f"\ncheckpoint: {successes}/{len(papers)} tasks succeeded")
    print("(success = read a section with the tools AND answered in valid JSON)")
    print(f"bar for staying with a 3B model: {BAR}/10")
    (HERE / "results").mkdir(exist_ok=True)
    suffix = "" if THINK is None else ("_think_on" if THINK else "_think_off")  # keep each setting's results apart
    (HERE / "results" / f"06_function_calling_by_hand{suffix}.json").write_text(
        json.dumps({"model": MODEL, "think": THINK, "rows": rows}, indent=1, default=str))
