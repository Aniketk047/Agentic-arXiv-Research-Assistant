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
| `get_paper_text(paper_id, query)` | Full-text retrieval beyond the abstract — see below. |

There is no routing logic in the code. Which tool fires, and when, is determined entirely by the tool descriptions in the schemas — the model reads them and decides. This makes description wording the primary engineering surface: the `compare_papers` description explicitly instructs the model to use it *instead of* repeated `fetch_paper` calls, because without that instruction the model defaults to calling `fetch_paper` twice.

Network calls are wrapped with exponential-backoff retries, and on final failure the tool returns an error string that tells the *model* the lookup failed and to say so rather than guess — handling the failure at the agent's semantic level, not just the network level.

### Full-text retrieval

`fetch_paper` only ever returns the abstract, so a question like "what optimizer did this paper use" was unanswerable — the answer isn't in an abstract. `get_paper_text` closes that gap: it downloads the PDF, splits it into sections by detecting headings structurally (a number followed by a short Title-Case phrase, plus a few known unnumbered headings like Abstract/References), chunks each section, and scores chunks by lexical overlap with the caller's query — returning only the top few, each tagged with its section heading.

Retrieval is lexical (term frequency + a heading-match bonus), not embedding-based. For a single paper's ~20-40 chunks that's cheap, deterministic, and needs no embedding model or extra API call; the tradeoff is it can miss a relevant section that doesn't share vocabulary with the question. The interface doesn't care — swapping in embeddings later wouldn't change the tool's signature.

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

### Current results: 6/8

The failures are the efficiency check flagging broad survey queries that fire 6-10 tool calls against a budget of 4. That's the suite working as intended — surfacing genuinely wasteful behaviour rather than rubber-stamping it. Whether 4 is the right threshold for open-ended survey questions (where searching from several angles may be legitimate) is an open design question, not a settled bug.

### Bugs the eval suite didn't catch

Two real bugs shipped past a green-ish eval run and were only found by reading the code, not by any automated check:

- `_download_pdf_text` called `raise_for_status()` *outside* the retry wrapper, so an HTTP error status on a PDF download (e.g. a transient 503 from arXiv's PDF servers) skipped the retry loop entirely — only connection-level errors were retried. Worse, the returned message claimed "failed after 3 attempts" when only one was ever made. Nothing in the suite exercises a failing network call, so a wrong message would have shipped straight to a user.
- The old-style-arXiv-ID regex added to close the pre-2007 grounding blind spot (below) used an unbounded `[a-z][a-z-]*` prefix, which also matched unrelated `word/1234567` shapes — a GitHub issue URL or an image asset path would have been misread as a citation. No case in the suite happens to contain that shape, so it passed clean.

Both are fixed now (verified with monkeypatched failures and targeted regex tests), but it's the same lesson as the citation-grounding bug from a different angle: things that never show up in the suite's specific eight queries are invisible to it. A passing suite means "consistent with what I thought to test," not "correct."

## Known limitations

**Full-text retrieval is lexical, not semantic.** `get_paper_text` scores chunks by term overlap with the query, so it can miss a relevant section that answers the question without sharing its vocabulary. Section detection is also heuristic (numbered headings plus a few known unnumbered ones), so unusually formatted PDFs may chunk oddly, and scanned/image-based PDFs with no extractable text return an explicit "couldn't extract text" message rather than silently failing.

**Grounding-check blind spots.** Pre-2007 IDs are now covered (anchored to arXiv's fixed list of legacy archive names), but papers cited by title only still aren't checked by the grounding assertion — the check is ID-based, so a fabricated title-only citation would slip through.

**Partial-failure in compare.** If one paper in a comparison fails to fetch, `compare_papers` still names which ID failed — the error string from `fetch_paper` is joined in alongside the successful results. But there's no structural marker distinguishing a success block from a failure block; the model (or a human) has to parse prose to tell which paper came back and which didn't.

## Next steps

- Embed chunks and rank by cosine similarity instead of lexical overlap, for queries that don't share vocabulary with the relevant section.
- Tune or make adaptive the efficiency budget, so multi-angle searching on broad queries isn't penalized the same as redundant calls on narrow ones.
- Extend the grounding check to catch fabricated title-only citations, not just IDs.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=your_key' > .env

python agent.py    # CLI
python eval.py     # eval suite
python app.py      # Gradio UI
```

## What I learned

[Write this in your own voice — 3-4 sentences. It's the part a reviewer reads for how you think. Honest angles you actually lived: tool *descriptions* turned out to be the real engineering surface, not the loop code; the grounding bug taught me prompt instructions aren't enforcement, evals are; a passing suite mostly told me my suite was too easy until I added checks that could fail; orchestrating a fixed model is a different discipline from training one.]
