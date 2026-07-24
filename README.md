# Agentic arXiv Research Assistant

A tool-using research agent built on Claude's tool-calling API. It searches arXiv,
fetches paper metadata, and compares papers across a multi-turn agentic loop —
plus a small eval harness and a Gradio chat UI on top.

## Architecture

- [`agent.py`](agent.py) — the agentic loop. Sends messages to Claude with the
  tool schemas, executes whichever tool the model calls, feeds the result back
  as a `tool_result`, and repeats until the model stops calling tools or a
  turn budget is hit.
- [`tools.py`](tools.py) — three tools (`search_papers`, `fetch_paper`,
  `compare_papers`) backed by the `arxiv` package, plus their JSON schemas.
- [`eval.py`](eval.py) — a handful of hand-written cases that check whether the
  right tool got called and whether expected keywords show up in the answer.
- [`app.py`](app.py) — a Gradio chat front end that also renders the tool-call
  trace so you can see what the agent actually did.

## Setup

```bash
pip install -r requirements.txt   # anthropic, arxiv, python-dotenv, gradio
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python app.py      # chat UI
python eval.py      # run the eval suite
```

## What I learned

A few things came out of building this that were more interesting than the
project itself.

**1. An agent loop is just a `while` loop with a stop condition, and that's
the hard part.** The mechanics of Claude's tool use — send messages, check
`stop_reason == "tool_use"`, execute, append a `tool_result`, loop — took
about twenty lines ([`agent.py:19-51`](agent.py#L19-L51)). The actual design
problem is deciding *when to stop the model from calling more tools*, not how
to call them. Without a turn cap the loop has no natural termination; with
too low a cap it truncates mid-task. I didn't appreciate how much of "agent
design" is really "budget design" until I watched it burn through calls.

**2. A passing eval isn't the same as correct behavior.** One of my eval
cases was "Who invented the telephone?" with the comment `# should decline or
redirect, not hallucinate papers`. The agent answered the question directly
(correctly, about Alexander Graham Bell) instead of redirecting, and the eval
marked it **PASS** anyway — because `expect_tools: []` and
`expect_keywords: []` are vacuously true for any answer. I'd written an
assertion that could never fail, which is worse than no assertion, because it
looks like coverage. The fix isn't more cases, it's checking that each
assertion can actually fail for the behavior it claims to test.

**3. Tool-calling agents will happily do more work than the task requires,
and nothing tells you unless you log it.** [`eval_results.json`](eval_results.json)
shows the "frequency-domain deepfake detection" query calling `search_papers`
three times and `fetch_paper` twice — eight tool calls for one question that
needed maybe two. The final answer was fine, so this would be invisible from
the outside. It only showed up because the eval harness happened to log the
full trace, not just the answer. Now I read tool-call counts as a signal
worth watching, the same way you'd watch latency or token cost.

**4. The most useful bug this project produced wasn't in the code I was
testing.** While pushing this repo, GitHub's secret scanning blocked the push
and pointed at `agent.py:7`. The line was
`Anthropic(api_key=os.getenv("sk-ant-api03-..."))` — the literal API key had
been passed as the *argument name* to `os.getenv()` instead of
`"ANTHROPIC_API_KEY"`, so the call silently returned `None` and would have
failed at runtime anyway. A typo-shaped mistake (wrong argument to the right
function) turned into a real credential leak, and the thing that caught it
was an external push-protection check, not a code review or a test. That's
the argument for defense in depth: the fix was one line, but I wouldn't have
caught it without a safety net that runs independently of my own judgment.
</content>
