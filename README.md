# JEV vs LLM: ranking research papers

Can a purpose-built scoring model rank arXiv papers as well as a general LLM,
for a fraction of the cost and latency? This repo runs both against the same
rubric, scores their top-5 picks against human labels, and charts the result.
You can rerun it on your own topic in a few minutes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-dark.svg">
  <img alt="Bar charts comparing LLM and JEV on ranking quality, latency and cost" src="docs/benchmark-light.svg">
</picture>

## Results so far

Topic: *looped transformers*, 10 arXiv candidates, graded 0–4 by a human.
**These results come from a single trial: one request per ranker.**

| | LLM (`anthropic/claude-sonnet-5.5`) | JEV (`typesafe/jev-1.13`) |
|---|---|---|
| Trials | 1 | 1 |
| Ranking quality (NDCG@5) | **1.000** | 0.978 |
| Latency | 5.34 s | **0.61 s** (~9× faster) |
| Cost per result | $0.0227 | **$0.00026** (~88× cheaper) |

Both rankers put the human's top paper first and agree on 4 of their 5 picks.
JEV's only miss was placing the grade-3 paper third instead of second.

**Why only one trial:** the published numbers are trial 1 of a 5-trial run.
In trials 2–5 every LLM request failed with HTTP 402 because the OpenRouter
account ran out of credit, so those trials were dropped. Across all 5 JEV
trials, the score was 0.978 every time and the top three picks never changed.
Only 4th and 5th place varied, always among papers graded 2. Median JEV
latency was 0.61 s. The run's `config.json` records this. `TRIALS` is set to 1 in
[benchmark_rankers.py](src/benchmark_rankers.py) to match.

**Read these numbers with care:**

- **Small sample.** One trial can't show how much either ranker varies from
  run to run, so treat the LLM figures in particular as a single data point.
- **Coarse labels.** 7 of the 10 papers share the same grade (2), so any top 5
  that includes the grade-4 and grade-3 papers scores close to 1.0. This test
  mainly checks whether a ranker finds those two papers.
- **One topic.** Results on other topics may differ. Please share yours.

## How the comparison works

Both rankers get identical input: the topic and each paper's title and
abstract. They score every paper on two rubrics, each from 0 to 4:

- **Relevance:** how directly the paper addresses the topic.
- **Learning value:** how useful it is as early reading for someone who knows
  standard Transformers but is new to the topic.

Rubrics and candidate text are defined once in
[benchmark_rankers.py](src/benchmark_rankers.py) and sent to:

| Ranker | OpenRouter endpoint | How scores come back |
|---|---|---|
| JEV | `/api/alpha/decisions` | One typed `score` answer per question |
| LLM | `/api/v1/chat/completions` | Strict JSON-schema structured output |

Every response is validated: all scores must be present, numeric and within
0–4, or the trial counts as invalid. The same rule then picks each ranker's
top 5 from its scores: drop papers with relevance below 2.5, then rank by
`0.4 × relevance + 0.6 × learning`. That top 5 is scored with **NDCG@5**
against the human grades in
[ranking_labels.json](src/ranking_labels.json). 1.0 means the
best possible order.

Each trial alternates which ranker goes first. Every run is saved to
`benchmark_results/run_<timestamp>/`, with the exact config, every request's
result (latency, cost, tokens, errors) and a summary.

## Run it yourself

You need [uv](https://docs.astral.sh/uv/) and an
[OpenRouter API key](https://openrouter.ai/keys) with credit on the account.
Each trial costs about $0.023, almost all of it the LLM call. If you see HTTP
402 errors, the account has run out of credit.

```bash
git clone https://github.com/sabmaverick1305/jev-vs-llm-paper-ranking.git
cd jev-vs-llm-paper-ranking
uv sync
cp .env.example .env    # then add your OPENROUTER_API_KEY
```

Run the benchmark, then redraw the charts:

```bash
uv run python src/benchmark_rankers.py
uv run python src/plot_benchmark.py
```

The charts pool every saved run that used the same candidates, rubrics and
labels as your newest run, so repeated runs add to the sample. `TRIALS` in
`benchmark_rankers.py` is 1; raise it (for example to 5) to measure how much
results vary between runs. To compare a
different LLM, set `BENCH_LLM_MODEL` in `.env` to any OpenRouter model that
supports structured outputs.

### Try your own topic

1. Fetch candidates from arXiv (10 works well):

   ```bash
   uv run python src/fetch_candidates.py "your topic" 10
   ```

2. Set `TOPIC` in `benchmark_rankers.py`. If your topic isn't a Transformer
   variant, also edit the reader description in the `learning` rubric.
3. Grade every paper in `src/ranking_labels.json` with a whole
   number from 0 (not useful) to 4 (best starting point), keyed by `paper_id`.
   Do this yourself, before looking at either ranker's output. Using the full
   range makes the comparison more informative.
4. Run the benchmark and plot as above.

## Project layout

```
src/
  benchmark_rankers.py               JEV vs LLM benchmark
  plot_benchmark.py                  Light/dark SVG charts → docs/
  fetch_candidates.py                Fetch arXiv candidates for a topic
  looped_transformer_candidates.json Candidate papers
  ranking_labels.json                Human grades (0–4)
  benchmark_results/                 One folder per run
docs/                                Generated charts
```

## References

- [arXiv API user manual](https://info.arxiv.org/help/api/user-manual.html)
- [OpenRouter structured outputs](https://openrouter.ai/docs/features/structured-outputs)
