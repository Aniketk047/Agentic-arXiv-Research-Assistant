import io
import re
import time
import requests
import arxiv
from pypdf import PdfReader

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

# ---------------------------------------------------------------------------
# Full-text retrieval (get_paper_text)
#
# A full arXiv paper is ~8k-20k tokens, too large to drop into the model's
# context on every call. So instead of returning the whole paper, we:
#   1. download the PDF and extract text,
#   2. split it into sections by detecting headings STRUCTURALLY (a number
#      followed by a short Title-Case phrase) plus a few known unnumbered
#      headings like Abstract/References — this generalizes across papers
#      instead of relying on a fixed list of section names,
#   3. chunk each section to a word budget,
#   4. score chunks by lexical overlap with the user's query and return only
#      the top-k, each tagged with its section heading.
#
# Retrieval here is lexical (query-term frequency + a heading-match bonus,
# length-normalized), not embedding-based. For a single paper (~20-40 chunks)
# this is cheap, deterministic, explainable, and needs no embedding model or
# extra API calls. The obvious upgrade for scale (many papers, fuzzy/semantic
# queries) is to embed chunks and query and rank by cosine similarity; the
# interface below would not change.
# ---------------------------------------------------------------------------

_KNOWN_HEADING = re.compile(
    r'^(Abstract|Introduction|Related Work|Background|References|'
    r'Acknowledge?ments|Appendix)\s*$', re.IGNORECASE)
_NUMBERED_HEADING = re.compile(r'^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z][^\n]{0,60})$')

_STOPWORDS = set(
    "the a an of to in and or for on with is are was were be been this that these "
    "those it its as at by from what which how does do about into paper papers "
    "arxiv id used use".split())

def _tokenize(s: str):
    return [w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in _STOPWORDS]

def _split_into_sections(text: str):
    lines = text.split("\n")
    idxs = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if _KNOWN_HEADING.match(s):
            idxs.append((i, s))
        else:
            m = _NUMBERED_HEADING.match(s)
            if m and len(s.split()) <= 8:   # a heading is a short title, not a sentence
                idxs.append((i, s))
    if not idxs:
        return [("Body", text)]
    sections = []
    if idxs[0][0] > 0:
        pre = "\n".join(lines[:idxs[0][0]]).strip()
        if pre:
            sections.append(("Preamble", pre))
    for j, (li, heading) in enumerate(idxs):
        end = idxs[j + 1][0] if j + 1 < len(idxs) else len(lines)
        body = "\n".join(lines[li:end]).strip()
        sections.append((heading, body))
    return sections

def _build_chunks(text: str, max_words: int = 350):
    chunks = []
    for heading, body in _split_into_sections(text):
        words = body.split()
        for i in range(0, len(words), max_words):
            chunks.append((heading, " ".join(words[i:i + max_words])))
    return chunks

def _score_chunks(query: str, chunks, top_k: int = 4):
    q_terms = _tokenize(query)
    if not q_terms:
        return chunks[:top_k]        # no usable query terms -> just return the opening chunks
    scored = []
    for heading, body in chunks:
        bt = _tokenize(body)
        if not bt:
            continue
        ht = set(_tokenize(heading))
        raw = sum(bt.count(t) + (3 if t in ht else 0) for t in q_terms)
        if raw > 0:
            scored.append((raw / (len(bt) ** 0.5), heading, body))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(h, b) for _, h, b in scored[:top_k]]

def _download_pdf_text(pdf_url: str) -> str:
    def _get():
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        return resp
    resp = _call_with_retries(_get)
    reader = PdfReader(io.BytesIO(resp.content))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def get_paper_text(paper_id: str, query: str = "") -> str:
    """Retrieve the most query-relevant chunks of a paper's FULL text (not just
    the abstract). Downloads the PDF, extracts and sections the text, and returns
    the top matching chunks with their section headings."""
    client = arxiv.Client()
    try:
        r = _call_with_retries(lambda: next(client.results(arxiv.Search(id_list=[paper_id]))))
    except StopIteration:
        return f"No paper with ID {paper_id}."
    except RETRYABLE_ERRORS as e:
        return _arxiv_error_message(f"fetch of {paper_id}", e)

    try:
        full_text = _download_pdf_text(r.pdf_url)
    except RETRYABLE_ERRORS as e:
        return _arxiv_error_message(f"PDF download of {paper_id}", e)
    except Exception as e:
        return (f"Could not extract text from the PDF for {paper_id} "
                f"({type(e).__name__}). The abstract is still available via fetch_paper.")

    if len(full_text.split()) < 50:
        return (f"Extracted almost no text from {paper_id} (likely a scanned or "
                f"image-based PDF). Fall back to fetch_paper for the abstract.")

    chunks = _build_chunks(full_text)
    top = _score_chunks(query, chunks, top_k=4)
    if not top:
        return (f"No chunks in {paper_id} matched the query '{query}'. "
                f"Try rephrasing, or use fetch_paper for the abstract.")

    header = f"Full-text excerpts from {paper_id} ({r.title}), most relevant to: '{query}'\n"
    body = "\n\n".join(f"[Section: {h}]\n{b}" for h, b in top)
    return header + "\n" + body


TOOL_FUNCTIONS = {
    "search_papers": search_papers,
    "fetch_paper": fetch_paper,
    "compare_papers": compare_papers,
    "get_paper_text": get_paper_text,
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
    {
        "name": "get_paper_text",
        "description": "Retrieve the most relevant chunks of a paper's FULL text (beyond the abstract). Use this when the user asks about methodology, datasets, experimental setup, hyperparameters, results, or any detail that would not appear in an abstract. Always pass the user's specific question as `query` so the right sections are selected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "arXiv ID, e.g. '1706.03762'"},
                "query": {"type": "string", "description": "The specific question to retrieve relevant sections for, e.g. 'what dataset and optimizer were used'"},
            },
            "required": ["paper_id", "query"],
        },
    },
]