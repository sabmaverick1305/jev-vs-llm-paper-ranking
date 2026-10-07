"""Fetch arXiv candidates for the benchmark.

Usage: uv run python src/fetch_candidates.py "your topic" [count]
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "looped_transformer_candidates.json"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def parse_papers(xml: bytes) -> list[dict]:
    papers = []
    for entry in ET.fromstring(xml).findall("a:entry", ATOM):
        source_url = entry.findtext("a:id", default="", namespaces=ATOM)
        if "/api/errors" in source_url:
            raise ValueError("arXiv rejected the query")
        if not source_url.startswith(("http://arxiv.org/abs/", "https://arxiv.org/abs/")):
            raise ValueError("Unexpected arXiv source URL")
        papers.append({
            "paper_id": source_url.split("/abs/", 1)[1],
            "title": " ".join(entry.findtext("a:title", default="", namespaces=ATOM).split()),
            "abstract": " ".join(entry.findtext("a:summary", default="", namespaces=ATOM).split()),
            "source_url": source_url.replace("http://", "https://", 1),
        })
    return papers


def search_arxiv(topic: str, limit: int) -> list[dict]:
    # Treat input as words, not arbitrary arXiv query syntax.
    words = re.findall(r"\w+", topic)[:12]
    if not words:
        raise ValueError("Topic must contain searchable words")
    response = httpx.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": " AND ".join(f'all:"{word}"' for word in words),
                "start": 0, "max_results": limit,
                "sortBy": "relevance", "sortOrder": "descending"},
        headers={"User-Agent": "JEVvsLLMBenchmark/0.1"},
        timeout=20,
    )
    response.raise_for_status()
    return parse_papers(response.content)


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    papers = search_arxiv(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 10)
    if not papers:
        raise SystemExit("No arXiv matches; try two or three specific words")
    OUTPUT.write_text(json.dumps(papers, indent=2) + "\n")
    print(f"Wrote {len(papers)} candidates to {OUTPUT}")
