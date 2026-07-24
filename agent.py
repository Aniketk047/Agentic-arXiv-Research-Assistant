import os
from anthropic import Anthropic
from dotenv import load_dotenv
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a research assistant for ML/AI papers on arXiv.

Approach:
- For topic questions, search_papers first, then fetch_paper for detail.
- For comparisons, gather all papers before answering.
- Cite papers by title and arXiv ID.
- If the tools return nothing useful, say so plainly. Never invent papers, authors, or findings.
- Be concise and technical. Assume the user knows ML fundamentals."""

def run_agent(user_message: str, max_turns: int = 6, verbose: bool = True):
    messages = [{"role": "user", "content": user_message}]
    trace = []

    for turn in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=2000,
            system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            final = "".join(b.text for b in resp.content if b.type == "text")
            return final, trace

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"[turn {turn}] {block.name}({block.input})")
                trace.append({"tool": block.name, "input": block.input})
                try:
                    output = TOOL_FUNCTIONS[block.name](**block.input)
                except Exception as e:
                    output = f"Tool error: {e}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})

    return "Hit max turns without a final answer.", trace

if __name__ == "__main__":
    answer, trace = run_agent("What are the main approaches to detecting GAN-generated images?")
    print("\n" + answer)
    print(f"\nTools used: {[t['tool'] for t in trace]}")