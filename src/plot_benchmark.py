"""Render benchmark results as light/dark SVG charts for the README.

Pools valid trials from every run in benchmark_results/ that used the same
benchmark setup (fingerprint) and human labels as the newest run.
"""
import json
import statistics
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "benchmark_results"
OUTPUT = HERE.parent / "docs"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "baseline": "#c3c2b7",
        "llm": "#2a78d6", "jev": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "baseline": "#383835",
        "llm": "#3987e5", "jev": "#d95926",
    },
}

PANELS = [
    ("ndcg_at_5", "Ranking quality", "NDCG@5 against human labels · higher is better",
     "mean", lambda v: f"{v:.3f}"),
    ("latency_ms", "Latency", "Median seconds per request · lower is better",
     "median", lambda v: f"{v / 1000:.2f}s"),
    ("cost_usd", "Cost", "Median USD per result, as reported by OpenRouter · lower is better",
     "median", lambda v: f"${v:.5f}"),
]

WIDTH, LEFT, RIGHT = 760, 132, 150
BAR, BAR_GAP, PANEL_GAP = 20, 10, 34


def load_rows():
    runs = sorted(RESULTS.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    if not runs:
        raise SystemExit("No runs in benchmark_results/; run benchmark_rankers.py first")
    newest = json.loads((runs[-1] / "config.json").read_text())
    if newest["labels"] is None:
        raise SystemExit("Newest run has no human labels; add ranking_labels.json and rerun")

    rows, used = [], 0
    for run in runs:
        config = json.loads((run / "config.json").read_text())
        if (config["fingerprint"], config["labels"]) != (newest["fingerprint"], newest["labels"]):
            continue
        used += 1
        with (run / "requests.jsonl").open() as file:
            rows += [row for row in map(json.loads, file) if row["valid"]]
    return newest["models"], rows, used


def summarize(rows):
    stats = {}
    for kind in ("llm", "jev"):
        valid = [row for row in rows if row["kind"] == kind]
        stats[kind] = {"n": len(valid)}
        for key, _, _, agg, _ in PANELS:
            values = [row[key] for row in valid if row.get(key) is not None]
            func = statistics.mean if agg == "mean" else statistics.median
            stats[kind][key] = func(values) if values else None
    return stats


def bar_path(x, y, width, height, radius=4):
    # Square at the baseline, 4px rounded data-end.
    if width <= radius:
        return f"M{x},{y}h{width}v{height}h{-width}z"
    w = width - radius
    return (f"M{x},{y}h{w}a{radius},{radius} 0 0 1 {radius},{radius}"
            f"v{height - 2 * radius}a{radius},{radius} 0 0 1 {-radius},{radius}h{-w}z")


def render(theme, models, stats, runs_used):
    c = THEMES[theme]
    plot = WIDTH - LEFT - RIGHT
    out, y = [], 28

    def text(x, y, body, fill, size=13, weight=400, anchor="start"):
        out.append(f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
                   f'font-weight="{weight}" text-anchor="{anchor}">{escape(body)}</text>')

    text(24, y, "JEV vs LLM: ranking arXiv papers for a learner", c["ink"], 17, 600)
    y += 22
    text(24, y, f'Topic "looped transformers" · 10 candidates · '
                f'{runs_used} run{"s" if runs_used != 1 else ""} pooled', c["ink2"])
    y += 26
    # Legend: always present for two series.
    x = 24
    for kind, label in (("llm", "LLM"), ("jev", "JEV")):
        out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{c[kind]}"/>')
        body = f"{label} · {models[kind]} · n={stats[kind]['n']} valid trial{'s' if stats[kind]['n'] != 1 else ''}"
        text(x + 18, y, body, c["ink2"])
        x += 18 + len(body) * 6.4 + 32
    y += 30

    for key, title, subtitle, _, fmt in PANELS:
        text(24, y, title, c["ink"], 14, 600)
        text(24, y + 18, subtitle, c["muted"], 12)
        y += 34
        values = [stats[k][key] for k in ("llm", "jev") if stats[k][key] is not None]
        top = 1.0 if key == "ndcg_at_5" else (max(values) if values else 1)
        top = top or 1
        out.append(f'<line x1="{LEFT}" y1="{y - 4}" x2="{LEFT}" '
                   f'y2="{y + 2 * BAR + BAR_GAP + 4}" stroke="{c["baseline"]}" stroke-width="1"/>')
        for kind, label in (("llm", "LLM"), ("jev", "JEV")):
            value = stats[kind][key]
            out.append(f'<rect x="{LEFT - 64}" y="{y + 6}" width="8" height="8" rx="2" fill="{c[kind]}"/>')
            text(LEFT - 50, y + 14, label, c["ink2"], 13, 500)
            if value is None:
                text(LEFT + 8, y + 14, "no valid trials", c["muted"], 12)
            else:
                # Keep tiny values visible as a sliver; the label carries the number.
                width = max(2, round(plot * value / top, 1))
                out.append(f'<path d="{bar_path(LEFT, y, width, BAR)}" fill="{c[kind]}"/>')
                text(LEFT + width + 8, y + 14, fmt(value), c["ink"], 13, 500)
            y += BAR + BAR_GAP
        y += PANEL_GAP - BAR_GAP

    text(24, y, "Generated by src/plot_benchmark.py from benchmark_results/.",
         c["muted"], 11)
    height = y + 20
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" font-family="system-ui, -apple-system, '
            f'\'Segoe UI\', sans-serif" role="img" aria-label="JEV vs LLM benchmark comparison">'
            f'<rect width="100%" height="100%" rx="12" fill="{c["surface"]}"/>'
            + "".join(out) + "</svg>\n")


if __name__ == "__main__":
    models, rows, runs_used = load_rows()
    stats = summarize(rows)
    OUTPUT.mkdir(exist_ok=True)
    for theme in THEMES:
        path = OUTPUT / f"benchmark-{theme}.svg"
        path.write_text(render(theme, models, stats, runs_used))
        print("Wrote", path)
    print(json.dumps(stats, indent=2))
