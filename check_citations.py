"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_CITE = re.compile(r"\[(\d+)\](?!\()")  # [3]  but not [3](link)
_REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_URL = re.compile(r"https?://\S+")


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources:
        return ["no sources in sources.json"]

    source_ns = set()
    by_n_url = {}
    seen_urls = {}
    for entry in sources:
        n = entry.get("n") if isinstance(entry, dict) else None
        if not isinstance(n, int):
            problems.append(f"source has non-integer n: {n!r}")
        else:
            source_ns.add(n)
            by_n_url[n] = entry.get("url")
        url = entry.get("url") if isinstance(entry, dict) else None
        if not (isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))):
            problems.append(f"source n={n!r} has invalid url: {url!r}")
            continue
        if url in seen_urls:
            problems.append(f"duplicate url between source n={seen_urls[url]!r} and n={n!r}: {url}")
        else:
            seen_urls[url] = n

    heading_matches = list(_REF_HEADING.finditer(report_text))
    if not heading_matches:
        problems.append("missing '## References' heading")
        return problems
    body = report_text[: heading_matches[-1].start()]
    references = report_text[heading_matches[-1].end():]

    # citations in the body only, skipping fenced/inline code spans
    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2:  # odd indexes are code spans/blocks: never touched
            continue
        for match in _CITE.finditer(segment):
            cited.add(int(match.group(1)))

    for n in sorted(cited - source_ns):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(source_ns - cited):
        problems.append(f"source [{n}] never cited")

    seen_ref_ns = set()
    for raw_line in references.splitlines():
        line = raw_line.strip()
        match = _REF_LINE.match(line)
        if not match:
            continue
        n = int(match.group(1))
        rest = match.group(2)
        if n in seen_ref_ns:
            problems.append(f"reference number [{n}] appears more than once in References")
        seen_ref_ns.add(n)
        if n not in source_ns:
            problems.append(f"reference [{n}] is not a known source")
        urls = _URL.findall(rest)
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL, found {len(urls)}")
        elif n in by_n_url and urls[0] != by_n_url[n]:
            problems.append(f"reference [{n}] url {urls[0]!r} does not match sources.json url {by_n_url[n]!r}")

    for n in sorted(source_ns - seen_ref_ns):
        problems.append(f"source [{n}] has no reference line in '## References'")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
