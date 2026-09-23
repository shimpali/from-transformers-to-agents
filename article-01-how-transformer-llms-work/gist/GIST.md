# Gist for this article

**Status: not yet published.**

The series convention (one Gist per article, not per snippet) is: paste [`how_transformer_llms_work.py`](how_transformer_llms_work.py)
into a new GitHub Gist, then embed that Gist's URL in the Medium draft so the code renders as a native,
syntax-highlighted, copy-pasteable block instead of a screenshot.

## How to publish it

1. Go to [gist.github.com](https://gist.github.com/).
2. Filename: `how_transformer_llms_work.py`.
3. Paste the contents of [`how_transformer_llms_work.py`](how_transformer_llms_work.py) in this folder.
4. Description: `From Transformers to Agents — Article 1: How Transformer LLMs Work`.
5. Create as a **public** Gist (Medium can only embed public Gists).
6. Copy the Gist URL and:
   - paste it into the Medium draft where the code block should render live, and
   - update the line below with the real URL.

## Published URL

`<paste the live gist.github.com URL here once created>`

## Why a local copy lives here too

The Gist is the thing Medium embeds, but this repo folder (and the notebook one level up) stays the
canonical, versioned source. If article code ever changes, edit `scripts/*.py` in the article folder,
regenerate this concatenated file, and update the Gist's content to match — the Gist is a mirror, not
the source of truth.
