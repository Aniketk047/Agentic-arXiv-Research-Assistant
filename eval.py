import json
import re
from agent import run_agent

# Legacy arXiv archives (pre-2007 IDs like hep-th/9901001, math.GT/0309136).
# Anchored to this fixed, finite list rather than any lowercase word — an
# open-ended [a-z][a-z-]* prefix also matches unrelated "word/1234567" shapes
# like GitHub issue URLs or asset paths.
_OLD_ARCHIVES = (
    r"astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|nlin|"
    r"nucl-ex|nucl-th|physics|quant-ph|math|cs|q-bio"
)
ARXIV_ID_RE = re.compile(
    rf"\b(\d{{4}}\.\d{{4,5}}|(?:{_OLD_ARCHIVES})(?:\.[A-Z]{{2,4}})?/\d{{7}})(?:v\d+)?\b"
)

def extract_ids(text: str) -> set:
    # arxiv.py's entry_id (used in tools.py) includes a version suffix like
    # "2101.09781v1" — strip it so tool output and model-cited IDs compare equal.
    return set(ARXIV_ID_RE.findall(text))

def trace_ids(trace: list) -> set:
    """Every arXiv ID that appeared anywhere in the tool inputs or outputs
    during the run — i.e. everything the agent actually had evidence for."""
    ids = set()
    for t in trace:
        ids |= extract_ids(str(t.get("input", "")))
        ids |= extract_ids(str(t.get("output", "")))
    return ids

def ungrounded_citations(answer: str, trace: list) -> set:
    """arXiv IDs cited in the answer that never showed up in any tool call."""
    return extract_ids(answer) - trace_ids(trace)

CASES = [
    {"q": "Find papers on frequency-domain deepfake detection",
     "expect_tools": ["search_papers"],
     "expect_keywords": ["frequency", "detection"]},
    {"q": "What is arXiv paper 2103.00020 about?",
     "expect_tools": ["fetch_paper"],
     "expect_keywords": ["CLIP", "image"]},
    {"q": "Compare 1706.03762 and 2005.14165",
     "expect_tools": ["compare_papers"],  # must use compare_papers specifically — fetch_paper x2 doesn't count
     "expect_keywords": ["transformer", "language"]},
    {"q": "Papers on Wav2Vec2 for speech tasks",
     "expect_tools": ["search_papers"],
     "expect_keywords": ["speech", "representation"]},
    {"q": "Who invented the telephone?",
     "expect_tools": [],
     "expect_keywords": []},  # should decline or redirect, not hallucinate papers
    {"q": "What's the arXiv ID for 'Attention Is All You Need'?",
     "expect_tools": ["search_papers", "fetch_paper"],
     "expect_keywords": []},  # tempts citing 1706.03762 from memory instead of verifying via tool call
    {"q": "What's the arXiv ID for the GPT-3 paper, 'Language Models are Few-Shot Learners'?",
     "expect_tools": ["search_papers", "fetch_paper"],
     "expect_keywords": []},  # tempts citing 2005.14165 from memory instead of verifying via tool call
    {"q": "What dataset and optimizer did the paper 1706.03762 use for training?",
     "expect_tools": ["get_paper_text"],
     "expect_keywords": ["Adam"]},  # abstract can't answer this -> must retrieve full text
]

MAX_TOOL_CALLS = 4

def score(case):
    answer, trace = run_agent(case["q"], verbose=False)
    used = [t["tool"] for t in trace]
    tool_ok = (not case["expect_tools"]) or any(t in used for t in case["expect_tools"])
    kw_ok = all(k.lower() in answer.lower() for k in case["expect_keywords"])
    ungrounded = ungrounded_citations(answer, trace)
    grounding_ok = not ungrounded
    efficient_ok = len(used) <= MAX_TOOL_CALLS
    return {"q": case["q"], "tools_used": used,
            "tool_ok": tool_ok, "keyword_ok": kw_ok, "grounding_ok": grounding_ok,
            "ungrounded_ids": sorted(ungrounded), "efficient_ok": efficient_ok,
            "passed": tool_ok and kw_ok and grounding_ok and efficient_ok, "answer": answer[:200]}

if __name__ == "__main__":
    results = [score(c) for c in CASES]
    passed = sum(r["passed"] for r in results)
    for r in results:
        flags = []
        if not r["grounding_ok"]:
            flags.append(f"UNGROUNDED IDS: {r['ungrounded_ids']}")
        if not r["efficient_ok"]:
            flags.append(f"INEFFICIENT: {len(r['tools_used'])} calls (max {MAX_TOOL_CALLS})")
        flag = "  " + "  ".join(flags) if flags else ""
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['q']}  tools={r['tools_used']}{flag}")
    print(f"\n{passed}/{len(results)} passed ({passed/len(results):.0%})")
    json.dump(results, open("eval_results.json", "w"), indent=2)