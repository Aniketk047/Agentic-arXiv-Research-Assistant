import time
import requests
import arxiv

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

# Network/HTTP failures worth retrying. arxiv.Client already retries some of
# these internally, but only up to its own default count and not on plain
# timeouts — this wrapper adds explicit, visible retries on top and is what
# decides when to finally give up and hand back an error string instead of
# letting the exception propagate into the agent loop.
RETRYABLE_ERRORS = (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError, requests.exceptions.RequestException)

def _call_with_retries(fn):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
    raise last_error

def _arxiv_error_message(action: str, e: Exception) -> str:
    return (f"arXiv {action} failed after {MAX_RETRIES} attempts due to a network/HTTP "
            f"error ({type(e).__name__}: {e}). This data is currently unavailable — "
            f"tell the user the lookup failed rather than guessing at paper details.")

def search_papers(query: str, max_results: int = 5) -> str:
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    try:
        results = _call_with_retries(lambda: list(client.results(search)))
    except RETRYABLE_ERRORS as e:
        return _arxiv_error_message("search", e)

    out = []
    for r in results:
        out.append(
            f"ID: {r.entry_id.split('/')[-1]}\n"
            f"Title: {r.title}\n"
            f"Authors: {', '.join(a.name for a in r.authors[:3])}\n"
            f"Published: {r.published.date()}\n"
            f"Abstract: {r.summary[:400]}...\n"
        )
    return "\n---\n".join(out) if out else "No papers found."

def fetch_paper(paper_id: str) -> str:
    client = arxiv.Client()
    try:
        r = _call_with_retries(lambda: next(client.results(arxiv.Search(id_list=[paper_id]))))
    except StopIteration:
        return f"No paper with ID {paper_id}."
    except RETRYABLE_ERRORS as e:
        return _arxiv_error_message(f"fetch of {paper_id}", e)
    return (f"Title: {r.title}\n"
            f"Authors: {', '.join(a.name for a in r.authors)}\n"
            f"Published: {r.published.date()}\n"
            f"Categories: {', '.join(r.categories)}\n\n"
            f"Abstract:\n{r.summary}")

def compare_papers(paper_ids: str) -> str:
    ids = [p.strip() for p in paper_ids.split(",")]
    return "\n\n=====\n\n".join(fetch_paper(i) for i in ids)

TOOL_FUNCTIONS = {
    "search_papers": search_papers,
    "fetch_paper": fetch_paper,
    "compare_papers": compare_papers,
}

TOOL_SCHEMAS = [
    {
        "name": "search_papers",
        "description": "Search arXiv for papers by topic or keyword. Returns IDs, titles, and abstracts. Use this first when the user asks about a research area rather than a specific paper.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'deepfake frequency detection'"},
                "max_results": {"type": "integer", "description": "How many papers to return (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_paper",
        "description": "Fetch full metadata and abstract for one arXiv paper by ID. Use when you already have an ID and need detail.",
        "input_schema": {
            "type": "object",
            "properties": {"paper_id": {"type": "string", "description": "arXiv ID, e.g. '2103.00020'"}},
            "required": ["paper_id"],
        },
    },
    {
        "name": "compare_papers",
        "description": "Fetch multiple arXiv papers in a single call for side-by-side comparison. ALWAYS use this instead of calling fetch_paper repeatedly when the user wants to compare, contrast, or relate two or more specific papers to each other.",
        "input_schema": {
            "type": "object",
            "properties": {"paper_ids": {"type": "string", "description": "Comma-separated arXiv IDs"}},
            "required": ["paper_ids"],
        },
    },
]