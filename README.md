# From Transformers to Agents

Code, notebooks, and article write-ups for my 11-part Medium series on DeepLearning.AI's language
model and agentic AI courses, written as a personal learning log rather than an expert take.

Each article gets its own numbered folder: a README with the full article text, a companion Jupyter
notebook, the standalone scripts the notebook is built from, the diagrams as SVGs, and the single
file that becomes that article's GitHub Gist for Medium's inline code embeds. The layout follows
[HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models),
adapted from numbered chapters to numbered articles.

`main.py` at the repo root automates the mechanical parts of getting an article from this repo onto
Medium: pushing its Gist and generating paste-ready HTML. See [Publishing with main.py](#publishing-with-mainpy)
below.

## Complete Series Plan (11 Articles)

Article titles link to each article's file in this repo for now. They'll be swapped for the
published Medium URLs as each one goes live.

---

## Section 1: Foundation (Understanding the Basics)

### Article 1: ["How Transformer LLMs Work"](article-01-how-transformer-llms-work/README.md)
- Tokenization, embeddings, attention, transformer blocks
- **Status:** Drafted, in review

### Article 2: ["Prompting Fundamentals for LLMs"](article-02-prompting-fundamentals-for-llms/README.md)
- How to instruct LLMs effectively, system prompts, prompt engineering basics
- **Status:** Not started

### Article 3: ["Function-Calling and Data Extraction with LLMs"](article-03-function-calling-and-data-extraction-with-llms/README.md)
- Making LLMs interface with external tools and data
- **Status:** Not started

---

## Section 2: Agentic AI Deep Dive (Core Patterns)

### Article 4: ["Building Agentic AI Workflows"](article-04-building-agentic-ai-workflows/README.md)
- What is agentic AI, degrees of autonomy, applications, task decomposition
- **Status:** Not started

### Article 5: ["The Four Design Patterns of Agentic AI"](article-05-the-four-design-patterns-of-agentic-ai/README.md)
- Overview of Reflection, Tool Use, Planning, Multi-Agent patterns
- **Status:** Not started

### Article 6: ["Reflection in AI: Teaching Agents to Critique Themselves"](article-06-reflection-in-ai/README.md)
- Why reflection beats direct generation, using external feedback, iterative improvement
- **Status:** Not started

### Article 7: ["Extending Agent Capabilities: Tools, APIs & MCP"](article-07-extending-agent-capabilities-tools-apis-mcp/README.md)
- Creating tools, connecting to external systems, code execution, Model Context Protocol
- **Status:** Not started

---

## Section 3: Orchestration (Connecting It All)

### Article 8: ["Advanced Autonomy: Planning and Multi-Agent Systems"](article-08-advanced-autonomy-planning-and-multi-agent-systems/README.md)
- Task planning, adaptive workflows, multi-agent orchestration, communication patterns
- **Status:** Not started

### Article 9: ["Evaluating and Optimizing Agents for Production"](article-09-evaluating-and-optimizing-agents-for-production/README.md)
- Building evals, error analysis, cost/latency optimization, production readiness
- **Status:** Not started

### Article 10: ["The A2A Protocol: Connecting AI Agents"](article-10-the-a2a-protocol/README.md)
- Agent discovery, client-server architecture, orchestrating multi-agent workflows across frameworks
- **Status:** Not started

---

## Section 4: Deployment & Implementation (Operating at Scale)

### Article 11: ["Prompting Multi-Agent Systems: A2A Workflows in Practice"](article-11-prompting-multi-agent-systems/README.md)
- Instructing orchestrated agents, system design for distributed systems, real-world patterns
- **Status:** Not started

---

## Series Overview

**Total Articles:** 11
**Structure:** Theory → Building → Scaling → Operating
**Target:** Technical practitioners, engineers, AI enthusiasts
**Approach:** Sharing learnings from DeepLearning.AI courses

### Course References

- [How Transformer LLMs Work](https://www.deeplearning.ai/courses/how-transformer-llms-work)
- [Function-Calling and Data Extraction with LLMs](https://www.deeplearning.ai/courses/function-calling-and-data-extraction-with-llms)
- [Agentic AI](https://www.deeplearning.ai/courses/agentic-ai)
- [A2A: The Agent2Agent Protocol](https://www.deeplearning.ai/courses/a2a-the-agent2agent-protocol)
- [AI Prompting for Everyone](https://www.deeplearning.ai/courses/ai-prompting-for-everyone)

## Folder layout

```
article-NN-article-slug/
├── README.md              full article text, images referenced via ./images/
├── article_slug.ipynb     companion notebook -- the scripts below, run in order
├── scripts/
│   ├── 01_....py          numbered, runnable, matches the article's flow
│   └── ...
├── images/
│   ├── 01-....svg         diagrams, numbered in reading order -- hand-authored, source of truth
│   ├── 01-....png         same diagram, auto-generated by main.py for Medium (SVG isn't supported there;
│   │                      gitignored, regenerated on demand, same as medium/ below)
│   └── ...
├── gist/
│   ├── article_slug.py    all scripts concatenated by main.py -- this is what gets pushed to the Gist
│   └── GIST.md            gist ID + published URL, kept in sync by `main.py --sync-gists`
└── medium/
    └── article_slug.medium.html   generated by main.py, open + select all + copy + paste into Medium
```

`gist/` and `medium/` are both generated by `main.py`; `medium/` isn't tracked in git (see
`.gitignore`) since it's regenerated on demand -- and neither are the generated `images/*.png`
files. Medium gets images by grabbing the actual pixels when you copy the opened HTML file, not by
fetching a URL, so the PNGs never need to be pushed to GitHub at all; they're purely a local,
disposable byproduct of running `main.py`.

## Publishing with main.py

`main.py` handles the two mechanical jobs of getting an article from this repo onto Medium. It takes
article folder names as arguments (or none, for every article), plus these flags:

- **`--sync-gists`**: builds each article's Gist from `scripts/*.py` and pushes it to GitHub through
  the Gist API. First run creates the gist; every run after that updates the same one in place
  (matched by the ID it stores in `gist/GIST.md`), so re-running after editing a script never creates
  duplicates. GitHub's Gist API still works fine for this, unlike Medium's own API, which stopped
  issuing new integration tokens years ago, so there's no equivalent automation for the Medium side.
- **Default, no flag needed**: converts each article's `README.md` into Medium-paste-ready HTML.
  First it converts every `images/*.svg` to a same-named `.png` (headless-Chromium render, pixel-for-
  pixel identical to the source diagram) since Medium doesn't accept SVG images at all -- the `.svg`
  stays untouched as the source of truth, the `.png` is a generated sibling, regenerated only when the
  `.svg` is newer. Image links then get rewritten to a local path relative to the generated HTML file
  itself, so opening it in a browser loads them straight off disk -- nothing needs to be pushed to
  GitHub first. Script links become real GitHub/Gist URLs, any tables get flattened into bullet lists
  (Medium doesn't support tables), the GitHub-only "In this folder" section gets cut, and the article's
  Gist URL gets dropped in near the end once one exists. Output: `<article>/medium/<slug>.medium.html`.

Run both together so the HTML always carries that run's fresh Gist URL:

```bash
python3 main.py --sync-gists
```

Or scope it to one article, or skip the gist sync and just regenerate the HTML:

```bash
python3 main.py article-01-how-transformer-llms-work --sync-gists
python3 main.py article-01-how-transformer-llms-work
```

### One-time setup for `--sync-gists`

1. `pip install -r requirements.txt` (covers `markdown`, `requests`, and `playwright`, along with
   the article code's own dependencies).
2. `playwright install chromium` -- one-time download of the browser `main.py` drives to convert
   SVGs to PNG. Only needed for that conversion step; everything else in the script works without it
   (SVG links just won't get rewritten to PNG, and Medium won't display them).
3. Create a token at [github.com/settings/tokens](https://github.com/settings/tokens): a classic
   token needs the **gist** scope; a fine-grained token needs the account-level **Gists** permission
   set to read and write.
4. Copy `.env.example` to `.env` and paste the token in as `GITHUB_TOKEN=...`. `.env` is gitignored,
   so it never gets committed.

### Getting the HTML onto Medium

Medium has no reliable publishing API left either, so this last step stays manual: open
`<article>/medium/<slug>.medium.html` in a browser, select all, copy, and paste into a new Medium
draft. Medium's editor reads the pasted formatting directly, headings and all.

One quirk: Medium only turns a URL into a live Gist embed when that URL is pasted directly into an
empty line by itself, not as part of a bigger paste. So after pasting the article, find the "Full
code for this article" line near the end, delete it, and paste that same gist URL again on its own to
get the live embed.

## Notebooks

Each article's `.ipynb` is assembled from that article's numbered scripts, in order, with a markdown
cell above each one matching the article's section headings. They're meant to be run top to bottom
in one sitting, for readers who'd rather execute the code than read it inline.

## Setup

```bash
pip install -r requirements.txt
```

Article 1's code runs real pretrained Hugging Face models (GPT-2, BERT) on CPU; nothing in the series
so far needs a GPU. See [Publishing with main.py](#publishing-with-mainpy) above for the extra
`.env` step needed only for `--sync-gists`.

## Notes

- Diagrams are hand-authored SVG with hardcoded hex colors (no CSS variables), so they render
  correctly both on GitHub and pasted into Medium.
- Code samples are original implementations inspired by the course/book, not reproduced from them --
  see the citation note at the bottom of each article's README.
