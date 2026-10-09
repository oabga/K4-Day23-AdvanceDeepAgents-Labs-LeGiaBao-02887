"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                delay = min(base * (2 ** attempt), cap) + random.uniform(0, base)
            time.sleep(delay)


def _clean(text):
    return " ".join((text or "").split())


def _http_get(url, *, params=None, timeout=20.0):
    """GET url; raise RetryableError on 429/5xx/transport errors; raise for any other bad status."""
    try:
        resp = httpx.get(url, params=params, timeout=timeout)
    except httpx.TransportError as exc:
        raise RetryableError(f"transport error: {exc}") from exc
    if resp.status_code == 429 or resp.status_code >= 500:
        retry_after = resp.headers.get("Retry-After")
        raise RetryableError(f"HTTP {resp.status_code}", retry_after=float(retry_after) if retry_after else None)
    resp.raise_for_status()
    return resp


# ---- TODO 2: arXiv ----
_arxiv_lock = threading.Lock()
_last_arxiv_call = [0.0]


def _arxiv_throttle():
    with _arxiv_lock:
        wait = _last_arxiv_call[0] + 3.0 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_arxiv_call[0] = time.monotonic()


def _parse_arxiv_atom(xml_text):
    root = ET.fromstring(xml_text)
    records = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        id_el = entry.find("atom:id", _ATOM_NS)
        if id_el is None or not (id_el.text or "").strip():
            continue
        raw_id = id_el.text.strip().rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)
        published_el = entry.find("atom:published", _ATOM_NS)
        title_el = entry.find("atom:title", _ATOM_NS)
        summary_el = entry.find("atom:summary", _ATOM_NS)
        records.append({
            "id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "published": (published_el.text or "")[:10] if published_el is not None else "",
            "title": _clean(title_el.text if title_el is not None else ""),
            "summary": _clean(summary_el.text if summary_el is not None else "")[:600],
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        terms = re.findall(r"[A-Za-z0-9-]+", query or "")
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{t}" for t in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max(1, min(int(max_results), 30)),
            "start": 0,
        }

        def _do():
            _arxiv_throttle()
            return _http_get(ARXIV_URL, params=params, timeout=20.0)

        resp = with_retry(_do, attempts=6, base=2.0, cap=60.0)
        records = _parse_arxiv_atom(resp.text)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
def _hf_record(item):
    paper = item.get("paper") or {}
    pid = paper.get("id")
    if not pid:
        return None
    summary = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
    return {
        "id": pid,
        "url": f"https://huggingface.co/papers/{pid}",
        "published": (paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
        "title": _clean(paper.get("title") or item.get("title") or ""),
        "summary": _clean(summary)[:600],
        "upvotes": paper.get("upvotes") or 0,
        "github": paper.get("githubRepo") or "",
        "stars": paper.get("githubStars") or 0,
    }


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date:
            params["date"] = date
        resp = with_retry(lambda: _http_get(HF_DAILY_URL, params=params, timeout=20.0))
        data = resp.json()
        records = [r for r in (_hf_record(item) for item in (data or [])) if r is not None]
        if keyword:
            kw = keyword.lower()
            records = [r for r in records if kw in (r["title"] + " " + r["summary"]).lower()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        params = {"q": query, "limit": max(1, min(int(limit), 50))}
        resp = with_retry(lambda: _http_get(HF_SEARCH_URL, params=params, timeout=20.0))
        data = resp.json()
        records = [r for r in (_hf_record(item) for item in (data or [])) if r is not None]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _exa_key():
    return (os.getenv("EXA_API_KEY") or "").strip()


def _redact(text, key):
    return text.replace(key, "<redacted>") if key else text


def _exa_rate_limited(result):
    blob = json.dumps(result).lower().replace(" ", "").replace("-", "").replace("_", "")
    return "ratelimit" in blob


def _exa_call(name, arguments):
    key = _exa_key()
    url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def _do():
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        except httpx.TransportError as exc:
            raise RetryableError(f"transport error: {_redact(str(exc), key)}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            raise RetryableError(f"HTTP {resp.status_code}", retry_after=float(retry_after) if retry_after else None)
        resp.raise_for_status()
        data = None
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if data is None:
            data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Exa error: {data['error']}")
        result = data.get("result") or {}
        if _exa_rate_limited(result):
            raise RetryableError("Exa rate limited (flag in result._meta)")
        texts = [c.get("text", "") for c in (result.get("content") or []) if c.get("type") == "text"]
        return "\n".join(t for t in texts if t)

    return with_retry(_do, attempts=6, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        objective = objective or f"find reliable, informative pages about: {query}"
        text = _exa_call("web_search_exa", {
            "query": query, "objective": objective, "numResults": max(1, min(int(num_results), 20)),
        }).strip()
        return text if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {_redact(f'{type(exc).__name__}: {exc}', _exa_key())}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        text = _exa_call("web_fetch_exa", {"urls": [url]}).strip()
        return text[:12000] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {_redact(f'{type(exc).__name__}: {exc}', _exa_key())}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
