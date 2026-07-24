import json
from agent import run_agent

CASES = [
    {"q": "Find papers on frequency-domain deepfake detection",
     "expect_tools": ["search_papers"],
     "expect_keywords": ["frequency", "detection"]},
    {"q": "What is arXiv paper 2103.00020 about?",
     "expect_tools": ["fetch_paper"],
     "expect_keywords": ["CLIP", "image"]},
    {"q": "Compare 1706.03762 and 2005.14165",
     "expect_tools": ["compare_papers", "fetch_paper"],
     "expect_keywords": ["transformer", "language"]},
    {"q": "Papers on Wav2Vec2 for speech tasks",
     "expect_tools": ["search_papers"],
     "expect_keywords": ["speech", "representation"]},
    {"q": "Who invented the telephone?",
     "expect_tools": [],
     "expect_keywords": []},  # should decline or redirect, not hallucinate papers
]

def score(case):
    answer, trace = run_agent(case["q"], verbose=False)
    used = [t["tool"] for t in trace]
    tool_ok = (not case["expect_tools"]) or any(t in used for t in case["expect_tools"])
    kw_ok = all(k.lower() in answer.lower() for k in case["expect_keywords"])
    return {"q": case["q"], "tools_used": used,
            "tool_ok": tool_ok, "keyword_ok": kw_ok,
            "passed": tool_ok and kw_ok, "answer": answer[:200]}

if __name__ == "__main__":
    results = [score(c) for c in CASES]
    passed = sum(r["passed"] for r in results)
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['q']}  tools={r['tools_used']}")
    print(f"\n{passed}/{len(results)} passed ({passed/len(results):.0%})")
    json.dump(results, open("eval_results.json", "w"), indent=2)