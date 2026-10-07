"""Pure parsing/validation helpers; these can be tested without API keys."""
import json
import xml.etree.ElementTree as ET

ATOM = {"a": "http://www.w3.org/2005/Atom"}


def parse_papers(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("a:entry", ATOM):
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


def attach_summaries(text: str, papers: list[dict]) -> list[dict]:
    # Fail explicitly on malformed/truncated output; no guessed JSON repair.
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {"summaries"}:
        raise ValueError("Expected a summaries object")
    rows = result["summaries"]
    if not isinstance(rows, list) or len(rows) != len(papers):
        raise ValueError("Expected exactly one summary per paper")
    allowed = {p["paper_id"] for p in papers}
    by_id = {}
    fields = {"paper_id", "problem", "approach", "reported_findings", "limitations"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Unexpected summary fields")
        if any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError("Summary fields must be nonempty strings")
        if row["paper_id"] not in allowed or row["paper_id"] in by_id:
            raise ValueError("Unknown or duplicate paper ID")
        by_id[row["paper_id"]] = row
    return [{**p, "summary": by_id[p["paper_id"]], "evidence_scope": "abstract_only"} for p in papers]
