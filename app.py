import gradio as gr
from agent import run_agent

def chat(message, history):
    answer, trace = run_agent(message, verbose=False)
    if trace:
        calls = "\n".join(f"- `{t['tool']}` {t['input']}" for t in trace)
        answer += f"\n\n<details><summary>Tool calls ({len(trace)})</summary>\n\n{calls}\n</details>"
    return answer

demo = gr.ChatInterface(
    fn=chat,
    title="Research Paper Agent",
    description="Agentic arXiv search using Claude tool use. Ask about topics, specific papers, or comparisons.",
    examples=["Find recent papers on RAG evaluation",
              "What is 1706.03762 about?",
              "Compare approaches to deepfake detection"],
)

if __name__ == "__main__":
    demo.launch()