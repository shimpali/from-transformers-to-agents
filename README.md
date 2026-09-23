# From Transformers to Agents

Code, notebooks, and article write-ups for my 11-part Medium series on DeepLearning.AI's language
model and agentic AI courses, written as a personal learning log rather than an expert take.

Each article gets its own numbered folder: a README with the full article text, a companion Jupyter
notebook, the standalone scripts the notebook is built from, the diagrams as SVGs, and the single
file that becomes that article's GitHub Gist for Medium's inline code embeds. The layout follows
[HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models),
adapted from numbered chapters to numbered articles.

## Series roadmap

| # | Article | Status | Folder |
|---|---|---|---|
| 1 | How Transformer LLMs Work | ✅ Drafted, in review | [`1-how-transformer-llms-work`](1-how-transformer-llms-work) |
| 2 | Prompting Fundamentals for LLMs | Not started | [`2-prompting-fundamentals-for-llms`](2-prompting-fundamentals-for-llms) |
| 3 | Function-Calling and Data Extraction with LLMs | Not started | [`3-function-calling-and-data-extraction-with-llms`](3-function-calling-and-data-extraction-with-llms) |
| 4 | Building Agentic AI Workflows | Not started | [`4-building-agentic-ai-workflows`](4-building-agentic-ai-workflows) |
| 5 | The Four Design Patterns of Agentic AI | Not started | [`5-the-four-design-patterns-of-agentic-ai`](5-the-four-design-patterns-of-agentic-ai) |
| 6 | Reflection in AI | Not started | [`6-reflection-in-ai`](6-reflection-in-ai) |
| 7 | Extending Agent Capabilities: Tools, APIs & MCP | Not started | [`7-extending-agent-capabilities-tools-apis-mcp`](7-extending-agent-capabilities-tools-apis-mcp) |
| 8 | Advanced Autonomy: Planning and Multi-Agent Systems | Not started | [`8-advanced-autonomy-planning-and-multi-agent-systems`](8-advanced-autonomy-planning-and-multi-agent-systems) |
| 9 | Evaluating and Optimizing Agents for Production | Not started | [`9-evaluating-and-optimizing-agents-for-production`](9-evaluating-and-optimizing-agents-for-production) |
| 10 | The A2A Protocol | Not started | [`10-the-a2a-protocol`](10-the-a2a-protocol) |
| 11 | Prompting Multi-Agent Systems | Not started | [`11-prompting-multi-agent-systems`](11-prompting-multi-agent-systems) |

Grouped into four sections: **Foundation** (1–3), **Agentic AI deep dive** (4–7), **Orchestration**
(8–10), **Deployment & implementation** (11).

## Folder layout

```
N-article-slug/
├── README.md              full article text, images referenced via ./images/
├── article_slug.ipynb     companion notebook -- the scripts below, run in order
├── scripts/
│   ├── 01_....py          numbered, runnable, matches the article's flow
│   └── ...
├── images/
│   ├── 01-....svg         diagrams, numbered in reading order
│   └── ...
└── gist/
    ├── article_slug.py    all scripts concatenated -- this is what gets pasted into the Gist
    └── GIST.md            publish checklist + the live gist.github.com URL once created
```

## Gists

Medium embeds code as GitHub Gists, not as plain code blocks, so each article gets exactly **one**
Gist (not one per snippet) built from that article's `gist/*.py` file. The workflow:

1. Finish the article's scripts under `scripts/`.
2. Concatenate them into `gist/<article_slug>.py` (numbered section comments, same order as the README).
3. Paste that file into a new **public** Gist on [gist.github.com](https://gist.github.com/).
4. Embed the Gist's URL in the Medium draft, and record it in that article's `gist/GIST.md`.

The repo copy under `scripts/` and `gist/` stays the source of truth; the Gist is a published mirror
of it. If code changes later, edit the repo first, then update the Gist to match.

## Notebooks

Each article's `.ipynb` is assembled from that article's numbered scripts, in order, with a markdown
cell above each one matching the article's section headings. They're meant to be run top to bottom
in one sitting, for readers who'd rather execute the code than read it inline.

## Setup

```bash
pip install -r requirements.txt
```

Article 1's code runs real pretrained Hugging Face models (GPT-2, BERT) on CPU; nothing in the series
so far needs a GPU.

## Notes

- Diagrams are hand-authored SVG with hardcoded hex colors (no CSS variables), so they render
  correctly both on GitHub and pasted into Medium.
- Code samples are original implementations inspired by the course/book, not reproduced from them --
  see the citation note at the bottom of each article's README.
