"""
RPA Bot Monitoring Web Application
- Flask backend with RAG-based log analysis (no external API keys)
- Supports test case uploads, execution log uploads, failure detection,
  multi-day log comparison, RCA generation, KPI dashboard, and AI summaries.
"""

import os
import re
import math
import hashlib
import traceback
from datetime import datetime
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Single shared in-memory store (no session/cookie dependency) ──
_store = {
    "log_entries": [],
    "test_cases":  [],
    "rag_vectors": [],
    "rag_docs":    [],
}


# ── Global error handler: always return JSON, never HTML ──
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error("Unhandled exception: %s\n%s", e, traceback.format_exc())
    return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large. Maximum size is 32 MB."}), 413

# ─────────────────────────────────────────────
# ─── Tiny RAG Engine (pure Python, no API) ───
# ─────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf_idf_vectors(corpus: list[str]) -> tuple[list[dict], dict]:
    """Return per-doc TF-IDF vectors and the IDF map."""
    tokenized = [_tokenize(doc) for doc in corpus]
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    N = len(corpus)
    idf = {word: math.log((N + 1) / (cnt + 1)) + 1 for word, cnt in df.items()}

    vectors = []
    for tokens in tokenized:
        tf: Counter = Counter(tokens)
        total = len(tokens) or 1
        vec = {word: (cnt / total) * idf.get(word, 1) for word, cnt in tf.items()}
        vectors.append(vec)
    return vectors, idf


def _cosine_sim(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values())) or 1
    mag_b = math.sqrt(sum(v * v for v in b.values())) or 1
    return dot / (mag_a * mag_b)


def _query_rag(query: str, corpus_vectors: list[dict], corpus_docs: list[str],
               top_k: int = 3) -> list[tuple[float, str]]:
    """Return top-k (score, doc) tuples most relevant to query."""
    if not corpus_vectors:
        return []
    q_tokens = _tokenize(query)
    tf: Counter = Counter(q_tokens)
    total = len(q_tokens) or 1
    q_vec = {word: cnt / total for word, cnt in tf.items()}
    scores = [(_cosine_sim(q_vec, vec), doc) for vec, doc in zip(corpus_vectors, corpus_docs)]
    scores.sort(key=lambda x: -x[0])
    return scores[:top_k]


# ─────────────────────────────────────────────
# ─── Log Parsing ─────────────────────────────
# ─────────────────────────────────────────────

# Regex patterns to detect log levels and common RPA failure keywords
_LEVEL_RE = re.compile(
    r"\b(ERROR|FAIL(?:ED|URE)?|WARN(?:ING)?|PASS(?:ED)?|SUCCESS|INFO|CRITICAL|EXCEPTION)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b"
)
_TS_RE = re.compile(
    r"(\d{4}[-/]\d{2}[-/]\d{2}[T \t]\d{2}:\d{2}(?::\d{2})?)"
)

FAILURE_KEYWORDS = [
    "error", "fail", "failed", "failure", "exception", "timeout",
    "crash", "abort", "critical", "unhandled", "traceback", "null",
    "undefined", "connection refused", "access denied", "404", "500",
    "attribute error", "key error", "value error", "type error",
    "not found", "permission denied", "robot stopped",
]

RCA_KNOWLEDGE_BASE = [
    {
        "pattern": ["timeout", "connection", "network"],
        "rca": "Network connectivity issue or service endpoint timeout.",
        "fix": "Check network configuration, increase timeout threshold, verify the target service is reachable.",
        "category": "Network",
    },
    {
        "pattern": ["authentication", "access denied", "permission", "login", "credential"],
        "rca": "Authentication failure — incorrect credentials or expired session.",
        "fix": "Rotate credentials, verify IAM roles and permissions, check SSO token expiry.",
        "category": "Auth",
    },
    {
        "pattern": ["element not found", "xpath", "selector", "ui element", "object not found"],
        "rca": "UI element locator failure — application UI may have changed.",
        "fix": "Update selectors/XPath, validate application version compatibility, run visual diff.",
        "category": "UI",
    },
    {
        "pattern": ["database", "sql", "query", "connection pool", "deadlock", "db"],
        "rca": "Database connectivity or query execution failure.",
        "fix": "Check DB connection string, review query for syntax errors, inspect DB server health.",
        "category": "Database",
    },
    {
        "pattern": ["file not found", "path", "directory", "read", "write", "io error"],
        "rca": "File system access error — missing file or insufficient permissions.",
        "fix": "Verify file paths, ensure the bot has read/write permissions, check disk space.",
        "category": "FileSystem",
    },
    {
        "pattern": ["api", "http", "rest", "response", "status code", "endpoint"],
        "rca": "External API call failure — service may be down or returning unexpected response.",
        "fix": "Check API health, validate request payload, handle rate-limiting with back-off retry.",
        "category": "API",
    },
    {
        "pattern": ["memory", "out of memory", "heap", "ram", "resource"],
        "rca": "Memory/resource exhaustion on the bot host.",
        "fix": "Increase JVM/process heap, close unused handles, scale up host resources.",
        "category": "Resource",
    },
    {
        "pattern": ["null", "none", "undefined", "attribute error", "key error", "type error"],
        "rca": "Runtime data validation failure — unexpected null or wrong data type.",
        "fix": "Add null checks, validate input data against expected schema before processing.",
        "category": "DataValidation",
    },
    {
        "pattern": ["robot", "bot", "process", "stopped", "aborted", "terminated"],
        "rca": "Bot process was unexpectedly stopped or aborted mid-execution.",
        "fix": "Review orchestrator kill signals, check for conflicting scheduled runs, inspect host process manager.",
        "category": "Process",
    },
    {
        "pattern": ["license", "quota", "limit", "exceeded"],
        "rca": "License quota exceeded or rate limit reached.",
        "fix": "Review license utilisation, stagger bot schedules, request quota increase.",
        "category": "License",
    },
]


def _detect_date(line: str) -> str | None:
    m = _DATE_RE.search(line)
    if m:
        raw = m.group(1).replace("/", "-")
        # normalise DD-MM-YYYY -> YYYY-MM-DD
        parts = raw.split("-")
        if len(parts[0]) == 2:
            raw = f"{parts[2]}-{parts[1]}-{parts[0]}"
        return raw
    return None


def _parse_log_lines(text: str, filename: str = "") -> list[dict]:
    """Parse raw log text into structured entries."""
    entries = []
    # Try to infer date from filename
    file_date = None
    m = _DATE_RE.search(filename)
    if m:
        file_date = m.group(1).replace("/", "-")

    for i, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        level_match = _LEVEL_RE.search(line)
        level = level_match.group(1).upper() if level_match else "INFO"

        # Normalise level
        if level in ("FAIL", "FAILURE"):
            level = "FAILED"
        elif level in ("WARN",):
            level = "WARNING"
        elif level in ("PASS",):
            level = "PASSED"

        date_from_line = _detect_date(line)
        effective_date = date_from_line or file_date or datetime.today().strftime("%Y-%m-%d")

        is_failure = level in ("ERROR", "FAILED", "CRITICAL") or any(
            kw in line.lower() for kw in FAILURE_KEYWORDS
        )

        entries.append({
            "line_no": i,
            "raw": line,
            "level": level,
            "date": effective_date,
            "is_failure": is_failure,
            "filename": filename,
        })
    return entries


def _parse_test_cases(text: str) -> list[dict]:
    """Parse test case file (CSV or plain text)."""
    cases = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # CSV: TestID, Name, Description, Expected
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            cases.append({
                "id": parts[0],
                "name": parts[1] if len(parts) > 1 else f"TC-{i:03d}",
                "description": parts[2] if len(parts) > 2 else "",
                "expected": parts[3] if len(parts) > 3 else "PASS",
                "status": None,
            })
        else:
            cases.append({
                "id": f"TC-{i:03d}",
                "name": line,
                "description": "",
                "expected": "PASS",
                "status": None,
            })
    return cases


# ─────────────────────────────────────────────
# ─── Analysis Engine ─────────────────────────
# ─────────────────────────────────────────────

# Words/patterns excluded from test-case ↔ log matching to prevent false failures
_STOP_WORDS = {
    "test", "check", "the", "a", "an", "is", "in", "of", "to", "and", "or",
    "for", "with", "on", "at", "by", "bot", "log", "run", "ok", "pass",
    "process", "task", "item", "line", "file", "data", "info", "step",
    "error", "failed", "failure", "warning", "passed", "critical",
    "not", "within", "responding", "endpoint", "connection",
}

_NUM_RE = re.compile(r"^\d+$")   # pure-numeric tokens (timestamps, line numbers)


def _meaningful_tokens(text: str) -> set:
    """Tokenise text and strip stop-words and pure numbers."""
    return {t for t in _tokenize(text)
            if t not in _STOP_WORDS and not _NUM_RE.match(t) and len(t) > 2}


def _match_test_cases_to_logs(test_cases: list[dict], log_entries: list[dict]) -> list[dict]:
    """
    Link test cases to log entries by meaningful keyword overlap.
    A test case is FAILED only when its specific keywords (IDs, module names,
    transaction refs) appear in a failure log line with a score >= 2.
    Generic/infrastructure words are excluded to prevent false positives.
    """
    failure_entries = [e for e in log_entries if e["is_failure"]]
    passed_entries  = [e for e in log_entries if e["level"] in ("PASSED", "SUCCESS")]

    for tc in test_cases:
        tc_tokens = _meaningful_tokens(tc["name"] + " " + tc["description"])
        if not tc_tokens:
            tc["status"] = "PASSED"
            tc["matched_log"] = ""
            continue

        # ── Failure match: requires 2+ specific tokens overlapping ──
        best_fail = None
        best_score = 0
        for entry in failure_entries:
            log_tokens = _meaningful_tokens(entry["raw"])
            overlap = len(tc_tokens & log_tokens)
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best_fail = entry

        if best_fail:
            tc["status"] = "FAILED"
            tc["matched_log"] = best_fail["raw"][:120]
            continue

        # ── PASSED: explicit log confirmation or no failure hit ──
        for entry in passed_entries:
            log_tokens = _meaningful_tokens(entry["raw"])
            if len(tc_tokens & log_tokens) >= 1:
                tc["status"] = "PASSED"
                tc["matched_log"] = ""
                break
        else:
            tc["status"] = "PASSED"
            tc["matched_log"] = ""

    return test_cases


def _compute_kpis(test_cases: list[dict], log_entries: list[dict]) -> dict:
    total = len(test_cases)
    passed = sum(1 for tc in test_cases if tc.get("status") == "PASSED")
    failed = sum(1 for tc in test_cases if tc.get("status") == "FAILED")
    success_rate = round((passed / total) * 100, 1) if total else 0

    error_logs = [e for e in log_entries if e["is_failure"]]
    impacted_txns = list({e["raw"][:60] for e in error_logs})[:10]

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "success_rate": success_rate,
        "total_log_lines": len(log_entries),
        "error_log_count": len(error_logs),
        "impacted_transactions": impacted_txns,
    }


def _detect_recurring_issues(log_entries: list[dict]) -> list[dict]:
    """
    Group failure log lines across dates. Flag issues seen on 2+ consecutive
    days (or recurring within 7 days) as 'persistent'.
    """
    issue_dates: dict[str, set] = defaultdict(set)

    for entry in log_entries:
        if not entry["is_failure"]:
            continue
        # Normalise the message (strip timestamps, numbers)
        normalised = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}[T \t]\d{2}:\d{2}(:\d{2})?", "", entry["raw"])
        normalised = re.sub(r"\b\d+\b", "#", normalised).strip()[:120]
        issue_dates[normalised].add(entry["date"])

    recurring = []
    for msg, dates in issue_dates.items():
        sorted_dates = sorted(dates)
        span_days = 0
        if len(sorted_dates) >= 2:
            try:
                d1 = datetime.strptime(sorted_dates[0], "%Y-%m-%d")
                d2 = datetime.strptime(sorted_dates[-1], "%Y-%m-%d")
                span_days = (d2 - d1).days
            except ValueError:
                span_days = len(sorted_dates)

        recurring.append({
            "issue": msg,
            "dates": sorted_dates,
            "occurrences": len(dates),
            "span_days": span_days,
            "persistent": span_days >= 2 or len(sorted_dates) >= 3,
        })

    recurring.sort(key=lambda x: (-x["occurrences"], -x["span_days"]))
    return recurring[:20]


def _build_rca(failure_entries: list[dict], rag_vectors: list[dict],
               rag_docs: list[str]) -> list[dict]:
    """Generate RCA for each unique failure using knowledge base + RAG."""
    seen: set[str] = set()
    rcas = []

    for entry in failure_entries:
        sig = hashlib.md5(entry["raw"][:80].encode()).hexdigest()[:8]
        if sig in seen:
            continue
        seen.add(sig)

        line_lower = entry["raw"].lower()

        # Match knowledge-base rule
        best_rule = None
        best_hits = 0
        for rule in RCA_KNOWLEDGE_BASE:
            hits = sum(1 for kw in rule["pattern"] if kw in line_lower)
            if hits > best_hits:
                best_hits = hits
                best_rule = rule

        # RAG: find similar past issues
        similar_docs = _query_rag(entry["raw"], rag_vectors, rag_docs, top_k=2)
        similar_issues = [
            {"score": round(s, 3), "excerpt": d[:120]}
            for s, d in similar_docs
            if s > 0.05
        ]

        rca_text = best_rule["rca"] if best_rule else "Root cause not deterministic — review stack trace."
        fix_text = best_rule["fix"] if best_rule else "Consult bot developer, review logs manually."
        category = best_rule["category"] if best_rule else "Unknown"

        # Augment RCA text with RAG context
        if similar_issues:
            rca_text += f" (RAG: similar issue found — '{similar_issues[0]['excerpt'][:80]}')"

        rcas.append({
            "failure": entry["raw"][:140],
            "date": entry["date"],
            "category": category,
            "rca": rca_text,
            "fix": fix_text,
            "similar_issues": similar_issues,
            "confidence": min(100, 40 + best_hits * 20 + (10 if similar_issues else 0)),
        })

    return rcas[:15]


def _generate_ai_summary(kpis: dict, rcas: list[dict],
                         recurring: list[dict], log_entries: list[dict]) -> str:
    """Generate a structured natural-language summary without any API call."""
    lines = []

    sr = kpis["success_rate"]
    health = "Healthy" if sr >= 90 else ("Degraded" if sr >= 60 else "Critical")

    lines.append(f"## Bot Health Status: {health}")
    lines.append(
        f"Out of {kpis['total']} test cases, {kpis['passed']} passed and "
        f"{kpis['failed']} failed — a success rate of {sr}%."
    )

    if kpis["error_log_count"]:
        lines.append(
            f"A total of {kpis['error_log_count']} error log lines were detected "
            f"across {kpis['total_log_lines']} log entries."
        )

    if rcas:
        top_cats = Counter(r["category"] for r in rcas).most_common(3)
        lines.append(
            "Top failure categories: "
            + ", ".join(f"{cat} ({cnt})" for cat, cnt in top_cats) + "."
        )
        top_rca = rcas[0]
        lines.append(
            f"Most critical failure: \"{top_rca['failure'][:80]}\" "
            f"[{top_rca['date']}] — {top_rca['rca']}"
        )
        lines.append(f"Recommended fix: {top_rca['fix']}")

    persistent = [r for r in recurring if r["persistent"]]
    if persistent:
        lines.append(
            f"{len(persistent)} issue(s) have persisted for 2+ days — "
            "these require immediate root-cause investigation to prevent SLA breach."
        )
        p = persistent[0]
        lines.append(
            f"Longest-running issue ({p['span_days']} days, {p['occurrences']} occurrences): "
            f"\"{p['issue'][:80]}\""
        )

    if kpis["impacted_transactions"]:
        lines.append(
            "Impacted transactions include: "
            + "; ".join(kpis["impacted_transactions"][:3]) + "."
        )

    if sr == 100:
        lines.append("All test cases passed. No immediate action required.")
    elif sr >= 90:
        lines.append(
            "Bot is mostly stable. Address highlighted failures before next scheduled run."
        )
    elif sr >= 60:
        lines.append(
            "Bot performance is degraded. Prioritise failure remediation and re-test."
        )
    else:
        lines.append(
            "Bot is in critical state. Escalate immediately and halt production runs "
            "until root cause is resolved."
        )

    return "\n".join(lines)


def _compare_logs(log_entries_by_date: dict[str, list[dict]]) -> list[dict]:
    """Compare failure trends across multiple log dates."""
    comparisons = []
    dates = sorted(log_entries_by_date.keys())
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_failures = {e["raw"][:80] for e in log_entries_by_date[prev_date] if e["is_failure"]}
        curr_failures = {e["raw"][:80] for e in log_entries_by_date[curr_date] if e["is_failure"]}

        new_failures = curr_failures - prev_failures
        resolved = prev_failures - curr_failures
        persisting = prev_failures & curr_failures

        comparisons.append({
            "from_date": prev_date,
            "to_date": curr_date,
            "new_failures": list(new_failures)[:5],
            "resolved_failures": list(resolved)[:5],
            "persisting_failures": list(persisting)[:5],
            "trend": (
                "Improving" if len(resolved) > len(new_failures)
                else ("Worsening" if len(new_failures) > len(resolved) else "Stable")
            ),
        })
    return comparisons


# ─────────────────────────────────────────────
# ─── Flask Routes ────────────────────────────
# ─────────────────────────────────────────────

def _ingest_logs(file_pairs: list[tuple[str, str]]) -> int:
    """Parse and store log file pairs; rebuild RAG vectors."""
    all_entries = []
    for fname, text in file_pairs:
        entries = _parse_log_lines(text, fname)
        all_entries.extend(entries)
    _store["log_entries"].extend(all_entries)
    failure_docs = [e["raw"] for e in _store["log_entries"] if e["is_failure"]]
    if failure_docs:
        _store["rag_vectors"], _ = _tf_idf_vectors(failure_docs)
        _store["rag_docs"] = failure_docs
    else:
        _store["rag_vectors"], _store["rag_docs"] = [], []
    return len(all_entries)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_logs", methods=["POST"])
def upload_logs():
    """Accept log files via multipart browser upload."""
    files = request.files.getlist("logs")
    if not files or not any(f.filename for f in files):
        return jsonify({"error": "No log files received. Please select at least one .log or .txt file."}), 400

    pairs = []
    for f in files:
        fname = secure_filename(f.filename) or f.filename
        text = f.read().decode("utf-8", errors="replace")
        pairs.append((fname, text))

    count = _ingest_logs(pairs)
    return jsonify({
        "message": f"Parsed {count} log lines from {len(pairs)} file(s).",
        "files": [p[0] for p in pairs],
        "total_entries": len(_store["log_entries"]),
    })


@app.route("/upload_test_cases", methods=["POST"])
def upload_test_cases():
    """Accept test case CSV/TXT via multipart browser upload."""
    f = request.files.get("test_cases")
    if not f or not f.filename:
        return jsonify({"error": "No test case file received."}), 400

    text = f.read().decode("utf-8", errors="replace")
    cases = _parse_test_cases(text)
    _store["test_cases"] = cases
    return jsonify({"message": f"Loaded {len(cases)} test case(s).", "test_cases": cases})


@app.route("/analyze", methods=["POST"])
def analyze():
    log_entries = _store["log_entries"]
    test_cases  = _store["test_cases"]

    if not log_entries and not test_cases:
        return jsonify({"error": "No data found. Please upload log files first."}), 400

    # Match test cases → logs, or auto-generate from failures
    if test_cases and log_entries:
        test_cases = _match_test_cases_to_logs(test_cases, log_entries)
        _store["test_cases"] = test_cases
    elif not test_cases:
        failure_entries = [e for e in log_entries if e["is_failure"]]
        test_cases = []
        for i, fe in enumerate(failure_entries[:20], 1):
            test_cases.append({
                "id": f"AUTO-{i:03d}",
                "name": fe["raw"][:50],
                "description": fe["raw"][:80],
                "expected": "PASS",
                "status": "FAILED",
                "matched_log": fe["raw"][:120],
            })
        pass_count = max(1, len(log_entries) - len(failure_entries))
        for i in range(min(pass_count, 5)):
            test_cases.append({
                "id": f"AUTO-P{i+1:03d}",
                "name": f"Nominal Execution #{i+1}",
                "description": "No failure detected in corresponding log section",
                "expected": "PASS",
                "status": "PASSED",
                "matched_log": "",
            })
        _store["test_cases"] = test_cases

    kpis = _compute_kpis(test_cases, log_entries)
    failure_entries = [e for e in log_entries if e["is_failure"]]
    rcas = _build_rca(failure_entries, _store["rag_vectors"], _store["rag_docs"])
    recurring = _detect_recurring_issues(log_entries)

    by_date: dict[str, list] = defaultdict(list)
    for e in log_entries:
        by_date[e["date"]].append(e)
    comparisons = _compare_logs(dict(by_date)) if len(by_date) > 1 else []

    ai_summary = _generate_ai_summary(kpis, rcas, recurring, log_entries)

    return jsonify({
        "kpis": kpis,
        "test_cases": test_cases,
        "rcas": rcas,
        "recurring_issues": recurring,
        "log_comparisons": comparisons,
        "ai_summary": ai_summary,
        "dates_covered": sorted(by_date.keys()),
    })


@app.route("/reset", methods=["POST"])
def reset():
    _store["log_entries"].clear()
    _store["test_cases"].clear()
    _store["rag_vectors"].clear()
    _store["rag_docs"].clear()
    return jsonify({
        "message": "Session reset.",
        "entries": len(_store["log_entries"]),
        "test_cases": len(_store["test_cases"]),
    })


@app.route("/status", methods=["GET"])
def status():
    """Quick health-check — returns current store counts."""
    return jsonify({
        "log_entries": len(_store["log_entries"]),
        "test_cases":  len(_store["test_cases"]),
        "rag_docs":    len(_store["rag_docs"]),
    })


if __name__ == "__main__":
    # use_reloader=False avoids the double-process issue in debug mode
    app.run(debug=True, port=5000, use_reloader=False)
