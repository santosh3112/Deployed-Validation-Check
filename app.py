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

# Stop-words excluded from matching to prevent false failures
_STOP_WORDS = {
    "test", "check", "the", "a", "an", "is", "in", "of", "to", "and", "or",
    "for", "with", "on", "at", "by", "bot", "log", "run", "ok", "pass",
    "process", "task", "item", "line", "file", "data", "info", "step",
    "error", "failed", "failure", "warning", "passed", "critical",
    "not", "within", "responding", "endpoint", "connection",
}

# RPA failure type classifier — maps keyword patterns to failure types
_FAILURE_TYPE_MAP = [
    (["timeout", "timed out", "not respond"],               "Timeout"),
    (["authentication", "access denied", "credential",
      "login", "permission", "unauthorized"],               "AuthFailure"),
    (["xpath", "element not found", "ui element",
      "selector", "object not found", "click"],             "UIFailure"),
    (["null", "none", "nullpointer", "attribute error",
      "key error", "value error", "type error"],            "DataError"),
    (["database", "sql", "db", "query", "connection pool",
      "deadlock"],                                          "DBFailure"),
    (["api", "http", "rest", "status code", "404", "500",
      "502", "503"],                                        "APIFailure"),
    (["file not found", "path", "io error", "directory"],   "FileSystemError"),
    (["memory", "out of memory", "heap", "oom"],            "ResourceExhaustion"),
    (["aborted", "stopped", "terminated", "crashed",
      "unhandled", "traceback"],                            "ProcessAbort"),
    (["exception", "error"],                                "RuntimeException"),
]

_NUM_RE = re.compile(r"^\d+$")


def _meaningful_tokens(text: str) -> set:
    """Tokenise text, strip stop-words and pure numbers."""
    return {t for t in _tokenize(text)
            if t not in _STOP_WORDS and not _NUM_RE.match(t) and len(t) > 2}


def _classify_failure_type(raw: str) -> str:
    """Return the most specific RPA failure type for a raw log line."""
    low = raw.lower()
    for patterns, ftype in _FAILURE_TYPE_MAP:
        if any(p in low for p in patterns):
            return ftype
    return "UnknownFailure"


def _match_test_cases_to_logs(test_cases: list[dict], log_entries: list[dict]) -> list[dict]:
    """
    Accurately match each test case against execution log entries.

    Decision logic:
      - FAILED  : 2+ specific keyword tokens overlap with a failure log line.
                  Records the exact matched log line, overlap score, failure type
                  and the RPA step where failure occurred.
      - PASSED  : Explicit PASSED/SUCCESS confirmation found in logs, OR no
                  failure match found (nominal execution assumed).

    Each test case gets enriched with:
      matched_log       — the exact log line that caused the failure
      match_score       — number of overlapping tokens (evidence strength)
      failure_type      — classified RPA failure category
      match_reason      — human-readable explanation of why it failed/passed
      log_level         — ERROR / FAILED / CRITICAL / WARNING of matched log
    """
    failure_entries = [e for e in log_entries if e["is_failure"]]
    passed_entries  = [e for e in log_entries if e["level"] in ("PASSED", "SUCCESS")]

    for tc in test_cases:
        tc_tokens = _meaningful_tokens(tc["name"] + " " + tc["description"])

        if not tc_tokens:
            tc.update({"status": "PASSED", "matched_log": "",
                       "match_score": 0, "failure_type": "",
                       "match_reason": "No specific keywords — defaulted to PASSED",
                       "log_level": ""})
            continue

        # ── 1. Find best-matching failure log line ──
        best_fail  = None
        best_score = 0
        best_overlap_tokens = set()

        for entry in failure_entries:
            log_tokens = _meaningful_tokens(entry["raw"])
            overlap_set = tc_tokens & log_tokens
            overlap = len(overlap_set)
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best_fail  = entry
                best_overlap_tokens = overlap_set

        if best_fail:
            ftype = _classify_failure_type(best_fail["raw"])
            tc.update({
                "status":       "FAILED",
                "matched_log":  best_fail["raw"][:160],
                "match_score":  best_score,
                "failure_type": ftype,
                "log_level":    best_fail["level"],
                "match_reason": (
                    f"Matched failure log via keywords: "
                    f"{', '.join(sorted(best_overlap_tokens)[:5])}. "
                    f"Failure type: {ftype}."
                ),
            })
            continue

        # ── 2. Look for explicit PASSED log confirmation ──
        confirmed_pass = None
        for entry in passed_entries:
            log_tokens = _meaningful_tokens(entry["raw"])
            if len(tc_tokens & log_tokens) >= 1:
                confirmed_pass = entry
                break

        tc.update({
            "status":       "PASSED",
            "matched_log":  "",
            "match_score":  0,
            "failure_type": "",
            "log_level":    confirmed_pass["level"] if confirmed_pass else "INFO",
            "match_reason": (
                f"Confirmed PASSED via log: \"{confirmed_pass['raw'][:80]}\""
                if confirmed_pass
                else "No failure match found — execution completed nominally."
            ),
        })

    return test_cases


def _compute_kpis(test_cases: list[dict], log_entries: list[dict]) -> dict:
    total  = len(test_cases)
    passed = sum(1 for tc in test_cases if tc.get("status") == "PASSED")
    failed = sum(1 for tc in test_cases if tc.get("status") == "FAILED")
    success_rate = round((passed / total) * 100, 1) if total else 0

    error_logs    = [e for e in log_entries if e["is_failure"]]
    impacted_txns = list({e["raw"][:60] for e in error_logs})[:10]

    # Per failure-type breakdown for KPI panel
    type_counts: dict = {}
    for tc in test_cases:
        if tc.get("status") == "FAILED" and tc.get("failure_type"):
            ft = tc["failure_type"]
            type_counts[ft] = type_counts.get(ft, 0) + 1

    return {
        "total":                total,
        "passed":               passed,
        "failed":               failed,
        "success_rate":         success_rate,
        "total_log_lines":      len(log_entries),
        "error_log_count":      len(error_logs),
        "impacted_transactions": impacted_txns,
        "failure_type_breakdown": type_counts,
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


# RPA-procedure fix playbook: maps failure type → step-by-step resolution
_RPA_FIX_PLAYBOOK = {
    "Timeout": [
        "1. Check network connectivity between the bot host and the target service.",
        "2. Increase the timeout threshold in the bot configuration (e.g. WaitTimeout, ElementTimeout).",
        "3. Add a retry loop with exponential back-off (e.g. 3 attempts, 5s / 15s / 30s delay).",
        "4. Verify the target application/service is healthy and not under load.",
        "5. Alert the infrastructure team if the service is persistently unreachable.",
    ],
    "AuthFailure": [
        "1. Rotate the bot credentials in the Credential Store / CyberArk / Secret Manager.",
        "2. Verify the bot service account is not locked or expired in Active Directory.",
        "3. Check SSO/OAuth token expiry — refresh tokens if required.",
        "4. Confirm IAM roles and permissions have not changed since last successful run.",
        "5. Re-run the bot with updated credentials and verify login succeeds.",
    ],
    "UIFailure": [
        "1. Verify the application version/build has not changed (UI regression).",
        "2. Update the XPath/CSS selector in the bot workflow to match the new UI layout.",
        "3. Add a 'wait for element' step before interacting to handle slow page loads.",
        "4. Take a screenshot at the point of failure for visual comparison.",
        "5. Re-run in attended mode to manually validate the selector before deploying.",
    ],
    "DataError": [
        "1. Add null/empty checks before passing data to downstream steps.",
        "2. Validate input data against the expected schema at the start of each transaction.",
        "3. Log the actual value received and compare against the expected data type.",
        "4. Implement an exception handler to move failed transactions to an error queue.",
        "5. Review upstream data source (DB, API, file) for missing or malformed records.",
    ],
    "DBFailure": [
        "1. Check the database connection string and credentials in the bot config.",
        "2. Verify the DB server is running and reachable from the bot host.",
        "3. Review the SQL query for syntax errors or schema changes.",
        "4. Check for connection pool exhaustion — increase pool size or add a retry.",
        "5. Review the DB server logs for deadlocks or maintenance windows.",
    ],
    "APIFailure": [
        "1. Verify the API endpoint URL and port are correct and the service is live.",
        "2. Check the HTTP status code — 401/403 = auth, 404 = endpoint changed, 5xx = server error.",
        "3. Validate the request payload matches the API contract (headers, content-type, body).",
        "4. Implement back-off retry for transient 5xx errors (max 3 retries).",
        "5. Contact the API owner if the endpoint has moved or the contract has changed.",
    ],
    "FileSystemError": [
        "1. Verify the file/folder path exists and is accessible from the bot host.",
        "2. Check that the bot service account has read/write permissions on the target path.",
        "3. Confirm disk space is available on the target drive.",
        "4. Check if the file is locked by another process.",
        "5. Add a pre-check step to validate file existence before the main workflow.",
    ],
    "ResourceExhaustion": [
        "1. Increase the JVM heap or process memory allocation for the bot runtime.",
        "2. Close unused file handles, browser windows, and DB connections after each transaction.",
        "3. Reduce the batch size per run to lower peak memory usage.",
        "4. Schedule the bot during off-peak hours to reduce resource contention.",
        "5. Monitor host CPU/RAM with an alerting rule — escalate if consistently >80%.",
    ],
    "ProcessAbort": [
        "1. Review the orchestrator logs for kill signals or conflicting scheduled jobs.",
        "2. Check for duplicate running instances — ensure only one instance runs at a time.",
        "3. Add a global exception handler to capture and log unhandled exceptions.",
        "4. Implement a bot health-check ping every N minutes to detect silent crashes.",
        "5. Escalate to the bot developer if a traceback/stack trace is present in the logs.",
    ],
    "RuntimeException": [
        "1. Capture the full stack trace from the log and identify the failing module and line.",
        "2. Reproduce the failure in a test environment with the same input data.",
        "3. Add try/except blocks around the failing code section with meaningful error logging.",
        "4. Validate all external dependencies (files, DB, APIs) are available before execution.",
        "5. Deploy the fix to a staging environment, run regression tests, then promote to production.",
    ],
}
_DEFAULT_PLAYBOOK = [
    "1. Review the full log file for the exact error message and stack trace.",
    "2. Isolate the failing step by running the bot in debug/attended mode.",
    "3. Check all external dependencies (services, files, credentials) are available.",
    "4. Add exception handling and retry logic around the failing step.",
    "5. Re-test and monitor for recurrence after the fix is deployed.",
]


def _generate_ai_summary(kpis: dict, rcas: list[dict],
                         recurring: list[dict], log_entries: list[dict],
                         test_cases: list[dict] = None) -> dict:
    """
    Generate a structured RPA-aware AI summary returned as a dict of sections.
    Each section is rendered separately in the UI for clear readability.
    """
    sr      = kpis["success_rate"]
    health  = "Healthy" if sr >= 90 else ("Degraded" if sr >= 60 else "Critical")
    failed_tcs  = [tc for tc in (test_cases or []) if tc.get("status") == "FAILED"]
    passed_tcs  = [tc for tc in (test_cases or []) if tc.get("status") == "PASSED"]
    persistent  = [r for r in recurring if r["persistent"]]

    # ── Section 1: Executive Overview ──
    overview_lines = [
        f"Bot Health: {health}  |  Success Rate: {sr}%  |  "
        f"{kpis['passed']} Passed / {kpis['failed']} Failed out of {kpis['total']} test cases.",
        f"{kpis['error_log_count']} error log lines detected across {kpis['total_log_lines']} total log entries.",
    ]
    if persistent:
        overview_lines.append(
            f"⚠ {len(persistent)} recurring issue(s) persisting for 2+ days — SLA risk."
        )

    # ── Section 2: What Happened (per failed test case) ──
    what_happened = []
    for tc in failed_tcs[:8]:
        what_happened.append({
            "tc_id":        tc.get("id", ""),
            "tc_name":      tc.get("name", ""),
            "failure_type": tc.get("failure_type", "UnknownFailure"),
            "match_reason": tc.get("match_reason", ""),
            "matched_log":  tc.get("matched_log", ""),
            "log_level":    tc.get("log_level", "ERROR"),
        })

    # ── Section 3: RPA Procedure Fix Steps (top unique failure types) ──
    seen_types: list[str] = []
    fix_steps = []
    for tc in failed_tcs:
        ft = tc.get("failure_type", "")
        if ft and ft not in seen_types:
            seen_types.append(ft)
            steps = _RPA_FIX_PLAYBOOK.get(ft, _DEFAULT_PLAYBOOK)
            fix_steps.append({"failure_type": ft, "steps": steps})
        if len(seen_types) >= 4:
            break

    # If no test case failures but RCA failures exist, fall back to RCA categories
    if not fix_steps and rcas:
        for rca in rcas[:3]:
            ft = rca.get("category", "")
            if ft and ft not in seen_types:
                seen_types.append(ft)
                steps = _RPA_FIX_PLAYBOOK.get(ft, _DEFAULT_PLAYBOOK)
                fix_steps.append({
                    "failure_type": ft,
                    "steps": steps,
                    "rca_note": rca.get("fix", ""),
                })

    # ── Section 4: Recommended Actions ──
    actions = []
    if sr == 100:
        actions.append("✅ All test cases passed. Monitor next scheduled run.")
    elif sr >= 90:
        actions.append("Bot is mostly stable. Fix highlighted failures before the next run.")
        actions.append("Review error logs and add retry logic for transient failures.")
    elif sr >= 60:
        actions.append("⚠ Bot performance is degraded. Prioritise remediation immediately.")
        actions.append("Pause non-critical bot schedules until failures are resolved.")
        actions.append("Raise a P2 incident ticket and assign to the RPA developer.")
    else:
        actions.append("🔴 CRITICAL: Halt all production runs immediately.")
        actions.append("Escalate to RPA Lead and Infrastructure team — raise a P1 incident.")
        actions.append("Do not re-enable until root cause is confirmed fixed and re-tested.")

    if persistent:
        p = persistent[0]
        actions.append(
            f"Persistent issue detected for {p['span_days']} days: "
            f'"{p["issue"][:70]}" — schedule a dedicated RCA session.'
        )

    if kpis["impacted_transactions"]:
        actions.append(
            f"Re-process {len(kpis['impacted_transactions'])} impacted transaction(s) "
            "after the fix is deployed."
        )

    # ── Section 5: Top RCA Summary ──
    rca_summary = []
    if rcas:
        top_cats = Counter(r["category"] for r in rcas).most_common(3)
        rca_summary.append(
            "Top failure categories: "
            + ", ".join(f"{cat} ({cnt})" for cat, cnt in top_cats) + "."
        )
        for rca in rcas[:3]:
            rca_summary.append(
                f"[{rca['category']}] {rca['failure'][:80]} → {rca['rca']}"
            )

    return {
        "health":        health,
        "success_rate":  sr,
        "overview":      "\n".join(overview_lines),
        "what_happened": what_happened,
        "fix_steps":     fix_steps,
        "actions":       actions,
        "rca_summary":   rca_summary,
    }


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

    ai_summary = _generate_ai_summary(kpis, rcas, recurring, log_entries, test_cases)

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
