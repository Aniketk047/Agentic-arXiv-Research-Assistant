# Agentic arXiv Research Assistant

An LLM agent that autonomously searches and synthesizes academic literature from arXiv. Ask a research question in natural language; the agent decides which tools to call, retrieves papers, and produces a cited answer.

Built with Python, the Anthropic API's tool-use interface, and Gradio.

## Demo

![demo](demo.png)


## Why an agent loop

A single API call can't answer research questions well: you don't know which papers are relevant until you've searched, and you don't know whether a paper is worth citing until you've read its abstract. Retrieval has to be interleaved with reasoning.

So the model runs in a loop. It receives tool schemas, decides which to invoke, gets results fed back as tool-result messages, and either calls more tools or produces a final answer — up to six turns.

In practice it does something more interesting than search-then-answer. On the query above it issued two parallel searches, read the results, fetched specific papers, then went *back* to search with better-informed queries ("GAN fingerprint frequency artifact") now that it knew the vocabulary of the field. That refinement step is emergent — nothing in the code tells it to search twice.

## Tools

| Tool | Purpose |
|---|---|
| `search_papers(query, max_results)` | Topic/keyword search. Returns IDs, titles, authors, truncated abstracts. |
| `fetch_paper(paper_id)` | Full metadata and abstract for one paper. |
| `compare_papers(paper_ids)` | Batch fetch for multi-paper comparison. |

There is no routing logic in the code. Which tool fires, and when, is determined entirely by the tool descriptions in the schemas — the model reads them and decides. This makes description wording the primary engineering surface: the `compare_papers` description explicitly instructs the model to use it *instead of* repeated `fetch_paper` calls, because without that instruction the model defaults to calling `fetch_paper` twice.

Network calls are wrapped with exponential-backoff retries, and on final failure the tool returns an error string that tells the *model* the lookup failed and to say so rather than guess — handling the failure at the agent's semantic level, not just the network level.

## Evaluation

LLM agents are non-deterministic, and prompt changes have non-obvious effects — rewording one tool description to fix a behaviour can silently break another. `eval.py` is an offline suite that makes changes measurable.

Each case scores four things independently:

- **Tool selection** — did the agent reach for the right tool? Case 3 is strict: comparing two papers must use `compare_papers`, not two `fetch_paper` calls.
- **Answer content** — did expected keywords appear?
- **Citation grounding** — see below.
- **Efficiency** — did the run stay within a call budget (max 4)?

A case passes only if all four hold. The suite also includes a negative case ("Who invented the telephone?") that passes only if the agent uses *no* tools — verifying it declines off-domain questions rather than inventing papers.

### Citation grounding — the eval that earned its place

The most useful check came from a real failure. In an early run the agent cited an arXiv ID that appeared nowhere in its tool trace — the model produced it from parametric memory, despite a system prompt telling it not to invent papers. Keyword-based checks missed this entirely.

So the suite extracts every arXiv ID from the answer and asserts each one appeared in some tool call's input or output. Two cases are designed specifically to bait the failure: they ask for the arXiv ID of famous papers ("Attention Is All You Need", the GPT-3 paper), tempting the model to emit `1706.03762` / `2005.14165` from memory. The fix — a system-prompt clause forbidding citation of any ID not seen in a tool result this conversation — is verified by these cases: the agent now searches to confirm the ID rather than reciting it.

The lesson: prompt instructions alone don't prevent hallucination, and you need an eval that fails on the real bug to know whether a fix held.

### Current results: 6/7

The one failure is the efficiency check flagging a broad survey query that fired 8 tool calls against a budget of 4. That's the suite working as intended — surfacing genuinely wasteful behaviour rather than rubber-stamping it. Whether 4 is the right threshold for open-ended survey questions (where searching from several angles may be legitimate) is an open design question, not a settled bug.

## Known limitations

**Abstract-only retrieval.** The tools return metadata and abstracts, never full text. The agent can tell you what a paper is about and how papers relate, but not what the authors actually did — no training setup, datasets, hyperparameters, or results. This is architectural, not a bug.

**Grounding-check blind spots.** The arXiv ID regex matches only the post-2007 `YYMM.NNNNN` format, so pre-2007 IDs (`hep-th/9901001`) and papers cited by title only aren't covered by the grounding assertion.

**Partial-failure in compare.** If one paper in a comparison fails to fetch, the run continues with partial data rather than signalling which paper was missing.

## Next steps

- `get_paper_text(paper_id)` — download the PDF and extract full text, with section-based chunking and query-relevant retrieval to stay within the context window (i.e. RAG over paper bodies). This is the main capability gap.
- Tune or make adaptive the efficiency budget, so multi-angle searching on broad queries isn't penalized the same as redundant calls on narrow ones.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install anthropic arxiv gradio python-dotenv requests
echo 'ANTHROPIC_API_KEY=your_key' > .env

python agent.py    # CLI
python eval.py     # eval suite
python app.py      # Gradio UI
```

## What I learned

[Write this in your own voice — 3-4 sentences. It's the part a reviewer reads for how you think. Honest angles you actually lived: tool *descriptions* turned out to be the real engineering surface, not the loop code; the grounding bug taught me prompt instructions aren't enforcement, evals are; a passing suite mostly told me my suite was too easy until I added checks that could fail; orchestrating a fixed model is a different discipline from training one.]
