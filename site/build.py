"""Build the project website (index.html at the repository root) from website.md.

    pip install markdown==3.7 pymdown-extensions==10.12
    python site/build.py

website.md is the single source of the page. The math ($...$ and $$...$$) is passed through untouched and rendered in
the browser by KaTeX; code blocks are coloured by highlight.js. GitHub Pages serves index.html from the main branch
root, so the figure paths in website.md (example*/figures/*.png) work as they are.
"""

from __future__ import annotations

import html
import pathlib
import re

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "website.md"
TEMPLATE = pathlib.Path(__file__).resolve().parent / "template.html"
OUTPUT = ROOT / "index.html"


def markdown_to_html(text: str) -> tuple[str, list[dict]]:
    """Markdown -> HTML body, plus the heading tree for the table of contents."""
    md = markdown.Markdown(extensions=[
        "tables",
        "toc",                              # ids on headings, and md.toc_tokens for the sidebar
        "pymdownx.arithmatex",              # keep $...$ / $$...$$ intact for KaTeX
        "pymdownx.superfences",             # fenced code blocks
        "pymdownx.highlight",               # <code class="language-xxx"> for highlight.js
    ], extension_configs={
        "pymdownx.arithmatex": {"generic": True},
        "pymdownx.highlight": {"use_pygments": False},
        "toc": {"toc_depth": "2-3"},
    })
    body = md.convert(text)
    return body, md.toc_tokens


def split_title(body: str) -> tuple[str, str]:
    """Take the first <h1> out of the body (the template shows it in the page header)."""
    match = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    if not match:
        return "", body
    return match.group(1), body[:match.start()] + body[match.end():]


def polish(body: str) -> str:
    """Small HTML fixes: figures with captions, scrollable tables, external links in a new tab."""
    # a paragraph that holds only an image becomes a <figure> whose caption is the alt text
    body = re.sub(r'<p>\s*<img alt="([^"]*)" src="([^"]*)"\s*/?>\s*</p>',
                  r'<figure><img src="\2" alt="\1" loading="lazy"><figcaption>\1</figcaption></figure>', body)
    # wide tables scroll sideways on small screens instead of breaking the layout
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    body = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', body)
    return body


def toc_html(tokens: list[dict]) -> str:
    """Nested <ul> of the level-2 headings (with their level-3 children) for the sidebar."""
    items = []
    for token in tokens:                     # level-1 title is skipped by toc_depth = "2-3"
        children = "".join(f'<li><a href="#{c["id"]}">{c["name"]}</a></li>' for c in token["children"])
        sub = f"<ul>{children}</ul>" if children else ""
        items.append(f'<li><a href="#{token["id"]}">{token["name"]}</a>{sub}</li>')
    return "<ul>" + "".join(items) + "</ul>"


def main():
    body, tokens = markdown_to_html(SOURCE.read_text(encoding="utf-8"))
    title, body = split_title(body)
    plain_title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
    page = (TEMPLATE.read_text(encoding="utf-8")
            .replace("{{title}}", html.escape(plain_title))
            .replace("{{title_html}}", title)
            .replace("{{toc}}", toc_html(tokens))
            .replace("{{body}}", polish(body)))
    OUTPUT.write_text(page, encoding="utf-8")
    (ROOT / ".nojekyll").touch()             # tell GitHub Pages to serve the files as they are
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
