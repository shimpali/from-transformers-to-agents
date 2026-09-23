#!/usr/bin/env python3
"""
main.py -- turn this repo's article folders into Medium-ready material.

Two jobs, one script:

  1. --sync-gists : create/update each article's GitHub Gist via the GitHub
     API, built by concatenating that article's scripts/*.py. GitHub's Gist
     API is alive and well documented (unlike Medium's, which stopped
     issuing new integration tokens -- see the "Medium" note below).
  2. (default)     : convert each article's README.md into Medium-paste-ready
     HTML: SVG diagrams converted to PNG, real image/gist URLs, tables
     flattened, GitHub-only sections cut.

Run both in one go -- sync first, then convert, so the Medium HTML always
carries that run's fresh Gist URL:

    python3 main.py --sync-gists
    python3 main.py article-01-how-transformer-llms-work --sync-gists
    python3 main.py                                    # convert only, every article
    python3 main.py article-01-how-transformer-llms-work --repo you/your-repo --branch main

--sync-gists needs a GitHub token with gist access. Put it in a .env file at
the repo root (never committed -- it's in .gitignore):

    GITHUB_TOKEN=ghp_...

Classic personal access token: needs the "gist" scope. Fine-grained token:
needs the account-level "Gists" permission set to read/write. Create one at
https://github.com/settings/tokens.

--- Medium ---
There's no reliable Medium publishing API left (Medium stopped issuing new
integration tokens), so getting the HTML onto Medium is still manual: open
the generated file in a browser, select all, copy, paste into a new Medium
draft. Medium's editor reads the pasted rich text directly.

What the Medium conversion does to each article's README.md:
  1. Converts every images/*.svg to a same-named .png (via a headless
     Chromium render, pixel-identical to the source diagram), because
     Medium does not accept SVG images at all -- only PNG/JPEG/GIF/WEBP.
     The .svg originals are left untouched in the repo; the .png is a
     generated sibling, only regenerated when the .svg is newer than it.
     Needs `playwright` (see "Requires" below) -- if it's missing, this
     step is skipped with a warning and the article's image links fall
     back to the raw .svg URL, which Medium will not render.
  2. Cuts everything from the "<!-- medium:cut -->" marker onward. Put that
     marker right before any GitHub-only section (this repo uses it before
     "## In this folder") so repo-navigation content never ends up in the
     Medium draft.
  3. Rewrites every relative link target:
       images/*.svg, *.png, *.jpg, ... -> local path, relative to the generated .medium.html
                                           (the browser loads these straight off disk when you
                                           open the file -- no need to push anything to GitHub
                                           first; Medium gets the actual pixels via copy/paste)
       scripts/*.py, once that article's gist is published -> the gist's url
       everything else (../README.md, unpublished scripts/, gist/, ...)
                                        -> github.com/<repo>/blob-or-tree/<branch>/...
  4. If the article's gist is published, adds a "Full code for this article:
     <bare gist URL>" line right before Sources. Medium turns a bare URL
     pasted on its own empty line into a live Gist embed -- that auto-embed
     only fires on a direct paste into an empty line, not partway through a
     bulk paste, so after pasting the article, select that line, delete it,
     and paste the same URL again by itself to trigger the embed.
  5. Flattens any remaining markdown tables into bullet lists, since Medium
     doesn't support real tables.
  6. Renders to HTML with the `markdown` library (fenced_code, sane_lists).

Output: <article-folder>/medium/<slug>.medium.html (slug = folder name with
the leading "article-NN-" stripped). Re-run any time the README, images, or
GIST.md changes -- this only ever reads README.md/images/scripts/GIST.md and
writes into images/ (the generated .png siblings), medium/, and gist/.

Known limitation this script can't fix: Medium's paste-import of code blocks
is inconsistent (sometimes the gray box survives, sometimes it degrades to
plain text) -- that's exactly why the one-gist-per-article convention exists.

Requires: pip install markdown requests playwright
          playwright install chromium   (one-time, downloads the browser
          playwright drives -- only needed for the SVG-to-PNG conversion;
          everything else in this script works without it)

--- Code layout ---
SvgToPng       -- SVG -> PNG conversion (headless Chromium).
GitHubGist     -- thin wrapper around the GitHub Gist API.
MediumConverter-- stateless markdown/HTML helpers + the shared templates.
Article        -- one article-NN-slug/ folder: title, gist state, link
                   resolution, and the convert()/sync_gist() pipelines.
SeriesRepo      -- the whole repo: discovers articles, detects owner/repo,
                   loads .env, and runs sync/convert across every article.
main()          -- CLI entry point, wires the above together.
"""
from __future__ import annotations

import argparse
import html
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("Missing dependency: pip install markdown")

try:
    import requests
except ImportError:
    requests = None  # only required for --sync-gists

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None  # only required for SVG -> PNG conversion


# ============================================================================
# SvgToPng -- headless-Chromium SVG -> PNG conversion
# ============================================================================

class SvgToPng:
    """Converts an article's images/*.svg diagrams to same-named .png
    siblings, since Medium doesn't accept SVG images at all. The .svg stays
    untouched as the source of truth; the .png is only regenerated when the
    .svg is newer than it. If playwright isn't installed, conversion is
    skipped with a one-time warning (printed once per script run)."""

    _warned = False

    @staticmethod
    def dimensions(svg_path: Path) -> tuple[float, float]:
        """Pull the pixel size out of an SVG's viewBox (preferred) or its
        width/height attributes, falling back to a sane default."""
        text = svg_path.read_text(encoding="utf-8")
        m = re.search(r'viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)"', text)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r'width="([\d.]+)[a-zA-Z%]*"[^>]*height="([\d.]+)[a-zA-Z%]*"', text)
        if m:
            return float(m.group(1)), float(m.group(2))
        return 1600.0, 900.0

    @classmethod
    def render(cls, svg_path: Path, png_path: Path, target_width: int = 1600) -> None:
        """Render one SVG to PNG via headless Chromium (pixel-faithful -- a
        real browser painting the real SVG -- unlike lighter SVG->raster
        libraries, which can miss markers/arrowheads and other features)."""
        w, h = cls.dimensions(svg_path)
        scale = max(target_width / w, 1.0)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(
                    viewport={"width": int(w), "height": int(h)},
                    device_scale_factor=scale,
                )
                page.goto(svg_path.resolve().as_uri())
                page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": w, "height": h})
                page.close()
            finally:
                browser.close()

    @classmethod
    def ensure_article_pngs(cls, article_dir: Path) -> None:
        """Make sure every images/*.svg in this article has an up-to-date
        sibling .png (regenerated only when the .svg is newer). The .svg
        originals are never touched."""
        images_dir = article_dir / "images"
        if not images_dir.is_dir():
            return
        svgs = sorted(images_dir.glob("*.svg"))
        if not svgs:
            return
        if sync_playwright is None:
            if not cls._warned:
                print("  ! playwright not installed -- SVGs can't be converted to PNG, "
                      "and Medium can't display SVGs.")
                print("    Fix: pip install playwright && playwright install chromium")
                cls._warned = True
            return
        for svg_path in svgs:
            png_path = svg_path.with_suffix(".png")
            if png_path.exists() and png_path.stat().st_mtime >= svg_path.stat().st_mtime:
                continue
            try:
                cls.render(svg_path, png_path)
                print(f"    converted {svg_path.name} -> {png_path.name}")
            except Exception as e:
                print(f"  ! couldn't convert {svg_path.name} to PNG: {e}")
                print("    Fix: playwright install chromium")


# ============================================================================
# GitHubGist -- create/update one Gist via the GitHub REST API
# ============================================================================

class GitHubGist:
    """Thin wrapper around the GitHub Gist API, authenticated with one
    token. One instance is shared across every article's sync."""

    API = "https://api.github.com"

    def __init__(self, token: str):
        self.token = token

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def create(self, description: str, filename: str, content: str) -> dict:
        resp = requests.post(
            f"{self.API}/gists",
            headers=self.headers,
            json={"description": description, "public": True, "files": {filename: {"content": content}}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def update(self, gist_id: str, description: str, filename: str, content: str) -> dict:
        resp = requests.patch(
            f"{self.API}/gists/{gist_id}",
            headers=self.headers,
            json={"description": description, "files": {filename: {"content": content}}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# MediumConverter -- markdown -> Medium-paste-ready HTML: shared templates,
# regexes, and the transforms that don't need per-article context
# ============================================================================

class MediumConverter:
    """Stateless helpers that turn README markdown into Medium-paste-ready
    HTML: cutting GitHub-only sections, flattening tables, and rendering.
    Link rewriting needs per-article context, so that lives on Article."""

    CUT_MARKER = "<!-- medium:cut -->"
    IMAGE_EXTS = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
    CODE_FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)
    LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
    TABLE_RE = re.compile(
        r"(?P<header>^\|.*\|[ \t]*\n)"
        r"(?P<sep>^\|[ \t:|-]+\|[ \t]*\n)"
        r"(?P<body>(?:^\|.*\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    HTML_TEMPLATE = """<!DOCTYPE html>
<!--
  Generated by main.py -- do not hand-edit, re-run the script instead.

  How to use: open this file in a browser, select all (Cmd/Ctrl+A), copy
  (Cmd/Ctrl+C), then paste into a new Medium draft (Cmd/Ctrl+V). Medium's
  editor reads the pasted formatting directly.

  If a "Full code for this article" line appears near the end, Medium's
  bulk paste will NOT auto-embed it as a live Gist widget -- that only
  triggers on pasting a bare URL directly into an empty line. So: select
  that line in the Medium draft, delete it, then paste the same URL again
  by itself to get the live embed.
-->
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    max-width: 700px;
    margin: 40px auto;
    padding: 0 20px;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 19px;
    line-height: 1.6;
    color: #1a1a1a;
  }}
  h1, h2, h3 {{ font-family: -apple-system, Helvetica, Arial, sans-serif; font-weight: 700; }}
  h1 {{ font-size: 34px; }}
  h2 {{ font-size: 26px; margin-top: 2em; }}
  h3 {{ font-size: 21px; margin-top: 1.5em; }}
  img {{ max-width: 100%; }}
  code {{ background: #f2f2f2; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #f2f2f2; padding: 14px; overflow-x: auto; border-radius: 4px; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 3px solid #ccc; margin-left: 0; padding-left: 1em; color: #555; }}
  a {{ color: #1a8917; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

    @classmethod
    def cut_at_marker(cls, text: str) -> str:
        if cls.CUT_MARKER in text:
            text = text.split(cls.CUT_MARKER, 1)[0]
        return text.rstrip() + "\n"

    @classmethod
    def process_outside_code(cls, text: str, fn) -> str:
        """Apply fn to every part of text that isn't inside a ``` fenced block ```."""
        parts = cls.CODE_FENCE_RE.split(text)
        return "".join(fn(p) if not p.startswith("```") else p for p in parts)

    @classmethod
    def flatten_tables(cls, text: str) -> str:
        def _flatten(m):
            header_cells = [c.strip() for c in m.group("header").strip().strip("|").split("|")]
            body_lines = [ln for ln in m.group("body").splitlines() if ln.strip()]
            out = []
            for line in body_lines:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                parts = []
                for label, val in zip(header_cells, cells):
                    val = val.replace("<br>", " ")
                    parts.append(f"**{label}:** {val}" if label else val)
                out.append("- " + " — ".join(parts))
            return "\n".join(out) + "\n"

        return cls.process_outside_code(text, lambda t: cls.TABLE_RE.sub(_flatten, t))

    @staticmethod
    def insert_gist_line(html_body: str, gist_url: str) -> str:
        para = f"<p>Full code for this article: {gist_url}</p>\n"
        marker = "<h3>Sources</h3>"
        if marker in html_body:
            return html_body.replace(marker, para + marker, 1)
        return html_body + para

    @staticmethod
    def render_html(text: str) -> str:
        return markdown.markdown(text, extensions=["fenced_code", "sane_lists"])

    @classmethod
    def wrap_html(cls, title: str, body: str) -> str:
        return cls.HTML_TEMPLATE.format(title=html.escape(title), body=body)


# ============================================================================
# Article -- one article-NN-slug/ folder
# ============================================================================

class Article:
    """One article-NN-slug/ folder: its README, scripts, images, Gist, and
    the Medium HTML generated from it."""

    GIST_MD_TEMPLATE = """# Gist for this article

**Status: published, kept in sync by `main.py --sync-gists`.**

One Gist per article (not per snippet), built by concatenating every file in
`scripts/`, in filename order. To update it after changing code, just run:

    python3 main.py {slug} --sync-gists

That rebuilds `gist/{filename}` from `scripts/*.py` and pushes it to this
same Gist (matched by the ID below) -- nothing to copy-paste by hand. The
Medium conversion then picks up the URL below automatically.

## Gist ID

{gist_id}

## Published URL

{html_url}
"""

    def __init__(self, path: Path, repo_root: Path):
        self.path = path
        self.repo_root = repo_root

    # -- identity -----------------------------------------------------------

    @property
    def slug(self) -> str:
        return self.path.name

    @property
    def short_slug(self) -> str:
        return re.sub(r"^article-\d+-", "", self.slug)

    @property
    def rel_dir(self) -> str:
        return self.path.relative_to(self.repo_root).as_posix()

    @property
    def medium_dir(self) -> str:
        return f"{self.rel_dir}/medium"

    @property
    def readme_path(self) -> Path:
        return self.path / "README.md"

    @property
    def gist_md_path(self) -> Path:
        return self.path / "gist" / "GIST.md"

    @property
    def title(self) -> str:
        if self.readme_path.exists():
            m = re.search(r"^#\s+(.+)$", self.readme_path.read_text(encoding="utf-8"), re.MULTILINE)
            if m:
                return m.group(1).strip()
        return self.slug

    # -- gist -----------------------------------------------------------------

    def gist_id(self) -> str | None:
        if not self.gist_md_path.exists():
            return None
        m = re.search(r"## Gist ID\s*\n+`?(\S+)`?", self.gist_md_path.read_text(encoding="utf-8"))
        return m.group(1) if m else None

    def gist_url(self) -> str | None:
        if not self.gist_md_path.exists():
            return None
        m = re.search(r"## Published URL\s*\n+`?(.+?)`?\s*(\n|$)", self.gist_md_path.read_text(encoding="utf-8"))
        if not m:
            return None
        url = m.group(1).strip()
        return url if url.startswith("http") else None

    def gist_status(self) -> str:
        if not self.gist_md_path.exists():
            return "no gist/GIST.md found"
        url = self.gist_url()
        return f"published: {url}" if url else "not yet published (placeholder in GIST.md)"

    def build_gist_source(self, repo: str, branch: str) -> tuple[str, str]:
        """Concatenate scripts/*.py into one file. Returns (filename,
        content); content is "" if there's nothing to sync."""
        scripts_dir = self.path / "scripts"
        scripts = sorted(scripts_dir.glob("*.py")) if scripts_dir.is_dir() else []
        filename = f"{self.short_slug.replace('-', '_')}.py"
        if not scripts:
            return filename, ""

        sep = "# " + "-" * 65
        lines = [
            f"# {self.title} -- all code from the article, concatenated",
            f"# https://github.com/{repo}/tree/{branch}/{self.slug}",
            "",
        ]
        for script in scripts:
            m = re.match(r"(\d+)_(.+)\.py$", script.name)
            label = f"{int(m.group(1))}. {m.group(2).replace('_', ' ').capitalize()}" if m else script.stem
            lines += [sep, f"# {label}", sep, script.read_text(encoding="utf-8").rstrip("\n"), ""]
        return filename, "\n".join(lines).rstrip() + "\n"

    def write_gist_md(self, filename: str, gist_id: str, html_url: str) -> None:
        content = self.GIST_MD_TEMPLATE.format(
            slug=self.slug, filename=filename, gist_id=gist_id, html_url=html_url,
        )
        (self.path / "gist" / "GIST.md").write_text(content, encoding="utf-8")

    def sync_gist(self, client: GitHubGist, repo: str, branch: str) -> None:
        filename, content = self.build_gist_source(repo, branch)
        if not content:
            print(f"  {self.slug}: skipped, no scripts/*.py found")
            return

        gist_dir = self.path / "gist"
        gist_dir.mkdir(exist_ok=True)
        (gist_dir / filename).write_text(content, encoding="utf-8")

        description = f"From Transformers to Agents -- {self.title}"
        existing_id = self.gist_id()
        try:
            if existing_id:
                data = client.update(existing_id, description, filename, content)
                action = "updated"
            else:
                data = client.create(description, filename, content)
                action = "created"
        except requests.exceptions.RequestException as e:
            print(f"  {self.slug}: FAILED -- {e}")
            return

        self.write_gist_md(filename, data["id"], data["html_url"])
        print(f"  {self.slug}: {action} -> {data['html_url']}")

    # -- link resolution --------------------------------------------------

    def resolve_link(self, target: str, repo: str, branch: str, gist_url: str | None) -> str:
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return target
        frag = ""
        if "#" in target:
            target, frag = target.split("#", 1)
            frag = "#" + frag
        combined = posixpath.normpath(posixpath.join(self.rel_dir, target))
        ext = posixpath.splitext(combined)[1].lower()
        if ext == ".svg":
            # Medium can't display SVGs -- point at the converted PNG
            # sibling instead, if SvgToPng managed to produce one.
            png_combined = combined[: -len(".svg")] + ".png"
            if (self.repo_root / png_combined).exists():
                combined = png_combined
                ext = ".png"
        if ext in MediumConverter.IMAGE_EXTS:
            # Local path, relative to where the generated .medium.html
            # lives -- the browser resolves this straight off disk, so
            # nothing needs to be hosted anywhere first.
            return posixpath.relpath(combined, self.medium_dir) + frag
        if gist_url and ext == ".py" and f"{self.rel_dir}/scripts/" in combined:
            return gist_url
        if ext == "":
            return f"https://github.com/{repo}/tree/{branch}/{combined}{frag}"
        return f"https://github.com/{repo}/blob/{branch}/{combined}{frag}"

    def rewrite_links(self, text: str, repo: str, branch: str, gist_url: str | None, log: list) -> str:
        def _sub(m):
            bang, label, target = m.groups()
            new_target = self.resolve_link(target, repo, branch, gist_url)
            if new_target != target:
                log.append((target, new_target))
            return f"{bang}[{label}]({new_target})"

        return MediumConverter.process_outside_code(text, lambda t: MediumConverter.LINK_RE.sub(_sub, t))

    # -- Medium conversion --------------------------------------------------

    def convert(self, repo: str, branch: str) -> None:
        if not self.readme_path.exists():
            print(f"  {self.slug}: skipped, no README.md")
            return

        SvgToPng.ensure_article_pngs(self.path)

        text = self.readme_path.read_text(encoding="utf-8")
        text = MediumConverter.cut_at_marker(text)

        gist_url = self.gist_url()
        log: list[tuple[str, str]] = []
        text = self.rewrite_links(text, repo, branch, gist_url, log)
        text = MediumConverter.flatten_tables(text)

        html_body = MediumConverter.render_html(text)
        if gist_url:
            html_body = MediumConverter.insert_gist_line(html_body, gist_url)

        html_doc = MediumConverter.wrap_html(self.title, html_body)

        out_dir = self.path / "medium"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{self.short_slug}.medium.html"
        out_path.write_text(html_doc, encoding="utf-8")

        print(f"  {self.slug}")
        print(f"    title:  {self.title}")
        print(f"    output: {out_path.relative_to(self.repo_root)}")
        print(f"    gist:   {self.gist_status()}")
        if log:
            print(f"    rewrote {len(log)} relative link(s):")
            for old, new in log:
                print(f"      {old}  ->  {new}")
        else:
            print("    no relative links found to rewrite")


# ============================================================================
# SeriesRepo -- the whole repo: discovers articles, detects owner/repo,
# loads .env, and runs sync/convert across every article
# ============================================================================

class SeriesRepo:
    """The whole from-transformers-to-agents repo."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def load_env(self) -> None:
        """Minimal .env loader: sets os.environ from KEY=VALUE lines,
        without overriding a variable already set in the real environment."""
        env_path = self.repo_root / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)

    def detect_repo(self) -> str | None:
        try:
            url = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.repo_root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return None
        m = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/]+?)(\.git)?$", url)
        return m.group("repo") if m else None

    def article(self, name: str) -> Article:
        return Article(self.repo_root / name, self.repo_root)

    def all_articles(self) -> list[Article]:
        dirs = sorted(p for p in self.repo_root.glob("article-*") if p.is_dir())
        return [Article(d, self.repo_root) for d in dirs]

    def sync_gists(self, articles: list[Article], repo: str, branch: str, token: str) -> None:
        client = GitHubGist(token)
        print(f"syncing gists to {repo}...\n")
        for article in articles:
            article.sync_gist(client, repo, branch)
        print()

    def convert(self, articles: list[Article], repo: str, branch: str) -> None:
        print(f"repo: {repo}   branch: {branch}\n")
        for article in articles:
            article.convert(repo, branch)
            print()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "articles", nargs="*",
        help="Article folder name(s). Omit to process every article-*/ folder.",
    )
    parser.add_argument("--repo", default=None, help="owner/repo on GitHub (auto-detected from 'git remote get-url origin' if omitted).")
    parser.add_argument("--branch", default="main", help="Branch for image/link URLs (default: main).")
    parser.add_argument("--sync-gists", action="store_true", help="Create/update each article's Gist via the GitHub API before converting. Needs GITHUB_TOKEN (.env or environment) with gist access.")
    args = parser.parse_args()

    series = SeriesRepo(Path(__file__).resolve().parent)
    series.load_env()

    repo = args.repo or series.detect_repo()
    if not repo:
        sys.exit(
            "Couldn't detect owner/repo from 'git remote get-url origin', and "
            "--repo wasn't given. Pass --repo yourname/your-repo."
        )

    if args.articles:
        articles = [series.article(name) for name in args.articles]
        for article in articles:
            if not article.path.is_dir():
                sys.exit(f"No such article folder: {article.rel_dir}")
    else:
        articles = series.all_articles()
        if not articles:
            sys.exit("No article-*/ folders found.")

    if args.sync_gists:
        if requests is None:
            sys.exit("Missing dependency for --sync-gists: pip install requests")
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            sys.exit(
                "GITHUB_TOKEN not set. Add GITHUB_TOKEN=ghp_... to a .env file at "
                "the repo root (needs 'gist' scope / account 'Gists' permission), "
                "or export it before running."
            )
        series.sync_gists(articles, repo, args.branch, token)

    series.convert(articles, repo, args.branch)


if __name__ == "__main__":
    main()
