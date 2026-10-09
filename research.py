"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import queue
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator

RUN_TIMEOUT_S = 25 * 60  # the sandbox can go idle-stale under the agent; never block forever on a hung call


def _invoke_with_timeout(agent, payload, config, timeout):
    """Run agent.invoke in a DAEMON thread so a hung underlying call (e.g. a stale sandbox connection after the
    sandbox went idle) cannot block the process forever. concurrent.futures.ThreadPoolExecutor is NOT used here:
    it joins its worker threads at interpreter exit, which would hang just the same. A daemon thread is simply
    abandoned by the interpreter instead."""
    box = queue.Queue(maxsize=1)

    def _run():
        try:
            box.put(("ok", agent.invoke(payload, config=config)))
        except Exception as exc:  # noqa: BLE001 - forward it to the caller instead of crashing a daemon thread
            box.put(("error", exc))

    threading.Thread(target=_run, daemon=True).start()
    try:
        status, value = box.get(timeout=timeout)
    except queue.Empty:
        raise RuntimeError(f"agent run exceeded {timeout}s and was abandoned") from None
    if status == "error":
        raise value
    return value


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    topic = (topic or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", topic).strip("-")
    slug = slug[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Research topic: {topic}\n\n"
        "Produce a citation-backed survey report on this topic, following your system prompt end to end: plan "
        "with write_todos, delegate sub-questions to researcher subagents in parallel with enough context, check "
        "their results, merge sources into sources.json (at least 3 of the 4 source families), write the report "
        "body following the required structure, run the finalizer script, run the validator script until it "
        "prints OK, and spot-check a few citations with the citation-checker subagent before finishing."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_calls = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0
    for msg in messages:
        if isinstance(msg, dict):
            calls = msg.get("tool_calls") or []
            usage = msg.get("usage_metadata")
        else:
            calls = getattr(msg, "tool_calls", None) or []
            usage = getattr(msg, "usage_metadata", None)
        for call in calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if not name:
                continue
            tool_calls[name] += 1
            if name == "task":
                subagent_calls += 1
        if usage:
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("agent did not produce a non-empty report.md")
    if not sources_bytes:
        raise RuntimeError("agent did not produce sources.json")
    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"sources.json is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json must be a non-empty JSON array")

    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    meta = summarize(messages, elapsed, model_name)
    meta.update({
        "topic": topic,
        "n_sources": len(sources),
        "source_families": sorted({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")}),
    })

    report_path = reports_dir / f"{slug}.md"
    report_path.write_text(report_bytes.decode("utf-8"), encoding="utf-8")
    (reports_dir / f"{slug}.sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2

    model = make_model()
    model_name = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or "unknown"
    start = time.monotonic()
    try:
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = _invoke_with_timeout(
                agent,
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                {"recursion_limit": 1000},
                RUN_TIMEOUT_S,
            )
            elapsed = time.monotonic() - start
            report_path = save_outputs(backend, topic, result["messages"], elapsed, model_name)
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - a broken run must exit 1, never leave a half-written report
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
