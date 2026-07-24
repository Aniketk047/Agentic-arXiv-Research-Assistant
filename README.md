# Agentic arXiv Research Assistant

An LLM agent that autonomously searches and synthesizes academic literature from arXiv. Ask a research question in natural language; the agent decides which tools to call, retrieves papers, and produces a cited answer.

Built with Python, the Anthropic API's tool-use interface, and Gradio.

## Demo

[screenshot or GIF here]


## Why an agent loop

A single API call can't answer research questions well: you don't know which papers are relevant until you've searched, and you don't know whether a paper is worth citing until you've read its abstract. Retrieval has to be interleaved with reasoning.

So the model runs in a loop. It receives tool schemas, decides which to invoke, gets results fed back as tool-result messages, and either calls more tools or produces a final answer — up to six turns.

In practice it does something more interesting than search-then-answer. On the query above it issued two parallel searches, read the results, fetched four specific papers, then went *back* to search with better-informed queries ("GAN fingerprint frequency artifact") now that it knew the vocabulary of the field. That refinement step is emergent — nothing in the code tells it to search twice.

## Tools

| Tool | Purpose |
|---|---|
| `search_papers(query, max_results)` | Topic/keyword search. Returns IDs, titles, authors, truncated abstracts. |
| `fetch_paper(paper_id)` | Full metadata and abstract for one paper. |
| `compare_papers(paper_ids)` | Batch fetch for multi-paper comparison. |

There is no routing logic in the code. Which tool fires, and when, is determined entirely by the tool descriptions in the schemas — the model reads them and decides. This makes description wording the primary engineering surface: the `compare_papers` description explicitly says to use it *instead of* repeated `fetch_paper` calls, because without that the model defaults to calling `fetch_paper` twice.

## Evaluation

LLM agents are non-deterministic, and prompt changes have non-obvious effects — rewording one tool description to fix a behaviour can silently break another. `evals.py` is an offline suite that makes changes measurable.

Each case scores two things independently:

- **Tool selection** — did the agent reason its way to the right tool? (measures decision-making)
- **Answer content** — did expected keywords appear? (measures output quality)

Separating these matters: an agent can pick the right tool and still write a bad answer, or reach a decent answer via the wrong path.

The suite includes a negative case ("Who invented the telephone?") that passes only if the agent uses *no* tools — verifying it declines off-domain questions rather than inventing papers.

Current: **5/5 passing**.

That number is not as good as it looks, and the suite is the weak part of this project rather than the strong part. Keyword matching is a crude proxy for correctness — an answer can contain the right words and still be wrong. And the tool check uses `any()`, so a case passes if the expected tool appears anywhere in the trace; one query fired eight calls where two would have sufficed and still passed. A suite that passes everything on the first run is measuring too little.

## Known limitations

**Abstract-only retrieval.** The tools return metadata and abstracts, never full text. The agent can tell you what a paper is about and how papers relate, but not what the authors actually did — no training setup, datasets, hyperparameters, or results. This is architectural, not a bug.

**Ungrounded citations.** In one run the agent cited an arXiv ID that appears nowhere in the tool trace — it came from the model's own knowledge, despite a system prompt instructing it not to invent papers. Prompt instructions alone are not sufficient to prevent this.

## Next steps

- `get_paper_text(paper_id)` — download the PDF and extract full text. Papers exceed the context window, so this requires section-based chunking and query-relevant retrieval.
- A citation-grounding eval that extracts every arXiv ID from the answer and asserts each appears in the tool trace. This would have caught the hallucinated citation above — an eval that fails on a real bug is worth more than five that pass.
- Tool-efficiency scoring, to penalize unnecessary calls.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install anthropic arxiv gradio python-dotenv
echo 'ANTHROPIC_API_KEY=your_key' > .env

python agent.py    # CLI
python evals.py    # eval suite
python app.py      # Gradio UI
```

## What I learned

[Write 3–4 sentences in your own voice. Suggested angles: that tool *descriptions* are the real engineering surface, not the code; that a 100% eval score told you more about your suite than your agent; that orchestrating a fixed model is a different discipline from training one.]
