"""
=============================================================================
RPA Bot Monitoring Web Application
=============================================================================
Purpose:
  A Flask-based web dashboard for monitoring RPA (Robotic Process Automation)
  bots.  It accepts execution log files and test case CSVs uploaded through
  the browser, then analyses them locally — no external API keys required.

Key Features:
  - Upload execution logs (.log / .txt) from multiple run dates
  - Upload test case definitions (CSV: ID, Name, Description, Expected)
  - Accurate test-case ↔ log matching to decide PASSED / FAILED
  - KPI dashboard: Total, Passed, Failed, Success Rate, Failure Types
  - Root Cause Analysis (RCA) using a knowledge base + local RAG engine
  - Recurring / persistent issue detection across multiple log dates
  - Multi-date log comparison (new / resolved / persisting failures)
  - RPA-procedure-aware AI summary with step-by-step fix playbook
  - Reset endpoint to clear all in-memory data between runs

RAG Engine:
  Uses pure Python TF-IDF + cosine similarity (stdlib only).
  No OpenAI / HuggingFace / external model needed.

Author   : RPA Monitoring Team
=============================================================================
"""

import os
import re
import math
import hashlib
import traceback
from datetime import datetime
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename   # sanitises uploaded filenames


# =============================================================================
# 1. APPLICATION SETUP
# =============================================================================

app = Flask(__name__)

# Limit uploaded file size to 32 MB to prevent memory exhaustion
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# Create an uploads directory if it does not already exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── In-memory data store ──────────────────────────────────────────────────
# A single shared dict that holds all runtime data for the current session.
# Using a module-level dict (instead of Flask sessions / cookies) ensures
# that every HTTP request sees the same state without cookie dependencies.
#
# Keys:
#   log_entries  — list of parsed log-line dicts (one per non-blank line)
#   test_cases   — list of test case dicts loaded from the CSV
#   rag_vectors  — list of TF-IDF weight dicts, one per failure log line
#   rag_docs     — corresponding raw failure log strings (the RAG corpus)
_store = {
    "log_entries": [],   # populated by /upload_logs
    "test_cases":  [],   # populated by /upload_test_cases
    "rag_vectors": [],   # rebuilt every time new log files are ingested
    "rag_docs":    [],   # mirror of the failure lines used to build vectors
}


# =============================================================================
# 2. GLOBAL ERROR HANDLERS
# =============================================================================

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Catch any unhandled Python exception inside a Flask view and return a
    clean JSON error instead of the default HTML 500 page.
    This prevents the browser from receiving unparseable HTML when the
    JavaScript fetch() call expects JSON.
    """
    app.logger.error("Unhandled exception: %s\n%s", e, traceback.format_exc())
    return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(_):
    """Return a readable JSON error when the uploaded file exceeds 32 MB."""
    return jsonify({"error": "File too large. Maximum size is 32 MB."}), 413


# =============================================================================
# 3. LOCAL RAG ENGINE  (TF-IDF + Cosine Similarity — no external API)
# =============================================================================
#
# How it works:
#   1. Every failure log line is treated as a "document" in the corpus.
#   2. _tf_idf_vectors() converts each document into a weighted word vector.
#   3. When a new failure occurs, _query_rag() converts the query into the
#      same vector space and finds the most similar past failures using
#      cosine similarity.
#   4. Similarity scores > 0.05 are surfaced as "similar past issues" in
#      the RCA cards, giving the analyst historical context.

def _tokenize(text: str) -> list[str]:
    """
    Convert raw text to a list of lowercase alphanumeric tokens.
    Example: "Connection timeout at 09:00" → ["connection", "timeout", "at", "09", "00"]
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf_idf_vectors(corpus: list[str]) -> tuple[list[dict], dict]:
    """
    Build TF-IDF weight vectors for every document in the corpus.

    TF  (Term Frequency)   = how often a word appears in THIS document.
    IDF (Inverse Doc Freq) = how rare a word is across ALL documents.
    TF-IDF weight          = TF × IDF  (high = important & rare)

    Returns:
        vectors — list of {word: weight} dicts, one per document
        idf     — the shared IDF lookup table (word → IDF score)
    """
    # Tokenise every document once
    tokenized = [_tokenize(doc) for doc in corpus]

    # Count how many documents contain each unique word (document frequency)
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))   # use set() so each word counts once per doc

    N = len(corpus)

    # IDF with +1 smoothing to avoid division-by-zero on unseen words
    idf = {word: math.log((N + 1) / (cnt + 1)) + 1 for word, cnt in df.items()}

    # Build one TF-IDF vector per document
    vectors = []
    for tokens in tokenized:
        tf: Counter = Counter(tokens)
        total = len(tokens) or 1          # avoid division-by-zero on empty doc
        vec = {word: (cnt / total) * idf.get(word, 1) for word, cnt in tf.items()}
        vectors.append(vec)

    return vectors, idf


def _cosine_sim(a: dict, b: dict) -> float:
    """
    Measure how similar two TF-IDF vectors are using cosine similarity.

    Cosine similarity = dot_product(a, b) / (|a| × |b|)
    Range: 0.0 (completely different) → 1.0 (identical direction).

    Only words present in BOTH vectors contribute to the dot product,
    making this efficient even for sparse high-dimensional vectors.
    """
    keys = set(a) & set(b)                             # shared vocabulary
    dot  = sum(a[k] * b[k] for k in keys)             # dot product
    mag_a = math.sqrt(sum(v * v for v in a.values())) or 1   # ||a||
    mag_b = math.sqrt(sum(v * v for v in b.values())) or 1   # ||b||
    return dot / (mag_a * mag_b)


def _query_rag(query: str, corpus_vectors: list[dict], corpus_docs: list[str],
               top_k: int = 3) -> list[tuple[float, str]]:
    """
    Find the top-k most similar past failure log lines for a given query.

    Steps:
      1. Tokenise the query and build a simple TF vector (no IDF needed for
         the query itself — keeps it fast).
      2. Compute cosine similarity between the query vector and every
         corpus vector.
      3. Return the top-k (score, document) pairs sorted by descending score.

    Args:
        query          — the new failure log line to search for
        corpus_vectors — pre-built TF-IDF vectors from _tf_idf_vectors()
        corpus_docs    — the original failure log strings (same order)
        top_k          — number of similar issues to return

    Returns:
        List of (similarity_score, log_line_text) tuples.
    """
    if not corpus_vectors:
        return []   # nothing to compare against yet

    # Build query vector (raw TF only — IDF not needed for single-query lookup)
    q_tokens = _tokenize(query)
    tf: Counter = Counter(q_tokens)
    total = len(q_tokens) or 1
    q_vec = {word: cnt / total for word, cnt in tf.items()}

    # Score every document and sort by highest similarity first
    scores = [(_cosine_sim(q_vec, vec), doc)
              for vec, doc in zip(corpus_vectors, corpus_docs)]
    scores.sort(key=lambda x: -x[0])
    return scores[:top_k]


# =============================================================================
# 4. LOG PARSING CONSTANTS & PATTERNS
# =============================================================================

# ── Regex: extract the log level from a log line ──────────────────────────
# Matches keywords like ERROR, FAILED, WARNING, PASSED, SUCCESS, INFO, etc.
# The (?:...) groups are non-capturing alternatives for variant spellings.
_LEVEL_RE = re.compile(
    r"\b(ERROR|FAIL(?:ED|URE)?|WARN(?:ING)?|PASS(?:ED)?|SUCCESS|INFO|CRITICAL|EXCEPTION)\b",
    re.IGNORECASE,
)

# ── Regex: extract a date from a string ───────────────────────────────────
# Matches both YYYY-MM-DD and DD-MM-YYYY (with - or / separators).
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b"
)

# ── Regex: extract a full timestamp (date + time) ─────────────────────────
# Used to strip timestamps during recurring-issue normalisation.
_TS_RE = re.compile(
    r"(\d{4}[-/]\d{2}[-/]\d{2}[T \t]\d{2}:\d{2}(?::\d{2})?)"
)

# ── Keywords that indicate a failure in a log line ────────────────────────
# A log line is marked as is_failure=True if its level is ERROR/FAILED/CRITICAL
# OR if any of these keywords appear anywhere in the line.
FAILURE_KEYWORDS = [
    "error", "fail", "failed", "failure", "exception", "timeout",
    "crash", "abort", "critical", "unhandled", "traceback", "null",
    "undefined", "connection refused", "access denied", "404", "500",
    "attribute error", "key error", "value error", "type error",
    "not found", "permission denied", "robot stopped",
]

# ── RCA Knowledge Base ─────────────────────────────────────────────────────
# A list of rules.  Each rule has:
#   pattern  — keywords to search for in the failure log line (any match wins)
#   rca      — plain-English root cause explanation
#   fix      — one-line recommended remediation
#   category — short label used in KPI badges and AI summary
#
# Rules are evaluated from top to bottom; the rule with the most keyword
# matches (best_hits) wins.  If no rule matches, a generic fallback is used.
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


# =============================================================================
# 5. LOG PARSING FUNCTIONS
# =============================================================================

def _detect_date(line: str) -> str | None:
    """
    Extract and normalise a date from a single log line.

    Handles two common formats:
      YYYY-MM-DD  (ISO format — returned as-is)
      DD-MM-YYYY  (European format — converted to ISO)

    Returns the date as a "YYYY-MM-DD" string, or None if no date is found.
    """
    m = _DATE_RE.search(line)
    if m:
        raw = m.group(1).replace("/", "-")   # normalise / separator to -
        parts = raw.split("-")
        # Detect DD-MM-YYYY by checking if the first segment is 2 digits
        if len(parts[0]) == 2:
            raw = f"{parts[2]}-{parts[1]}-{parts[0]}"   # flip to YYYY-MM-DD
        return raw
    return None


def _parse_log_lines(text: str, filename: str = "") -> list[dict]:
    """
    Parse the raw text of a log file into a list of structured log-entry dicts.

    Each entry contains:
      line_no    — 1-based line number within the file
      raw        — the original (stripped) log line text
      level      — normalised level: ERROR / FAILED / WARNING / PASSED / INFO / CRITICAL
      date       — YYYY-MM-DD string (from the line, filename, or today)
      is_failure — True if the line represents a failure event
      filename   — the source filename (used for multi-file identification)

    Date resolution priority:
      1. Date embedded in the log line itself (most accurate)
      2. Date found in the filename (e.g. "2024-01-20_bot.log")
      3. Today's date as a last resort

    Empty lines are skipped.
    """
    entries = []

    # Step 1: Try to extract a date from the filename as a fallback
    file_date = None
    m = _DATE_RE.search(filename)
    if m:
        file_date = m.group(1).replace("/", "-")

    # Step 2: Process each line
    for i, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue   # skip blank lines

        # Detect the log level (e.g. ERROR, FAILED, PASSED)
        level_match = _LEVEL_RE.search(line)
        level = level_match.group(1).upper() if level_match else "INFO"

        # Normalise variant spellings to canonical values
        if level in ("FAIL", "FAILURE"):
            level = "FAILED"
        elif level == "WARN":
            level = "WARNING"
        elif level == "PASS":
            level = "PASSED"

        # Resolve the effective date for this line
        date_from_line = _detect_date(line)
        effective_date = (
            date_from_line                              # best: from log line
            or file_date                                # fallback: from filename
            or datetime.today().strftime("%Y-%m-%d")   # last resort: today
        )

        # Mark the line as a failure if level OR keyword indicates it
        is_failure = level in ("ERROR", "FAILED", "CRITICAL") or any(
            kw in line.lower() for kw in FAILURE_KEYWORDS
        )

        entries.append({
            "line_no":   i,
            "raw":       line,
            "level":     level,
            "date":      effective_date,
            "is_failure": is_failure,
            "filename":  filename,
        })

    return entries


def _parse_test_cases(text: str) -> list[dict]:
    """
    Parse a test case file (CSV or plain text) into a list of test case dicts.

    Expected CSV format (one test case per line):
        TestID, Name, Description, Expected
    Example:
        TC-001, SAP Connection Test, Verify SAP endpoint is reachable, PASS

    Lines starting with '#' are treated as comments and skipped.
    Plain text lines (no commas) are accepted as the test case name only.

    Each returned dict has:
      id          — test case identifier (e.g. "TC-001")
      name        — short test name
      description — longer description (used in keyword matching)
      expected    — expected outcome ("PASS" by default)
      status      — None initially; set to "PASSED" or "FAILED" after analysis
    """
    cases = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue   # skip comments and blank lines

        # Split by comma — expect at least ID and Name
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            cases.append({
                "id":          parts[0],
                "name":        parts[1] if len(parts) > 1 else f"TC-{i:03d}",
                "description": parts[2] if len(parts) > 2 else "",
                "expected":    parts[3] if len(parts) > 3 else "PASS",
                "status":      None,   # filled in by _match_test_cases_to_logs()
            })
        else:
            # Fallback: treat the whole line as a test case name
            cases.append({
                "id":          f"TC-{i:03d}",
                "name":        line,
                "description": "",
                "expected":    "PASS",
                "status":      None,
            })
    return cases


# =============================================================================
# 6. ANALYSIS ENGINE
# =============================================================================

# ── Stop-words for keyword matching ──────────────────────────────────────
# These generic words appear in almost every test case name AND every log
# line, so they add no discriminative power and cause false-positive
# failure matches.  They are stripped before any token overlap is measured.
_STOP_WORDS = {
    "test", "check", "the", "a", "an", "is", "in", "of", "to", "and", "or",
    "for", "with", "on", "at", "by", "bot", "log", "run", "ok", "pass",
    "process", "task", "item", "line", "file", "data", "info", "step",
    "error", "failed", "failure", "warning", "passed", "critical",
    "not", "within", "responding", "endpoint", "connection",
}

# ── Failure type classifier ───────────────────────────────────────────────
# An ordered list of (keyword_patterns, failure_type_label) tuples.
# The first matching entry wins (most-specific patterns listed first).
# This converts a raw error log line into a human-readable failure category
# that is shown in the KPI table and the AI summary playbook.
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

# ── Regex to detect pure-numeric tokens (e.g. timestamps, line numbers) ──
# "2024", "09", "00", "142" are not meaningful for semantic matching.
_NUM_RE = re.compile(r"^\d+$")


def _meaningful_tokens(text: str) -> set:
    """
    Convert text into a set of meaningful keyword tokens.

    Filtering pipeline:
      1. Tokenise: split on non-alphanumeric, lowercase
      2. Remove stop-words (generic words with no discriminative value)
      3. Remove pure-numeric tokens (timestamps, line numbers)
      4. Remove very short tokens (length ≤ 2 — usually noise)

    Example:
      "2024-07-17 09:00 ERROR SAP timeout in InvoiceParser"
      → {"sap", "timeout", "invoiceparser"}
    """
    return {
        t for t in _tokenize(text)
        if t not in _STOP_WORDS        # remove generic words
        and not _NUM_RE.match(t)       # remove pure numbers
        and len(t) > 2                 # remove very short tokens
    }


def _classify_failure_type(raw: str) -> str:
    """
    Classify a failure log line into one of the predefined RPA failure types.

    Scans _FAILURE_TYPE_MAP from top to bottom and returns the label of the
    first matching entry.  Returns "UnknownFailure" if nothing matches.

    This label is used to:
      - Display a coloured badge in the KPI evidence table
      - Select the correct step-by-step fix playbook in the AI summary
    """
    low = raw.lower()
    for patterns, ftype in _FAILURE_TYPE_MAP:
        if any(p in low for p in patterns):
            return ftype
    return "UnknownFailure"


def _match_test_cases_to_logs(test_cases: list[dict], log_entries: list[dict]) -> list[dict]:
    """
    Compare each test case against the execution log entries and decide
    whether the test case PASSED or FAILED.

    Decision Algorithm:
    ─────────────────
    For each test case:

      Step 1 — Failure detection:
        Extract meaningful tokens from the test case name + description.
        For every failure log line, compute the overlap between test tokens
        and log tokens.  If 2+ tokens match, the test case is marked FAILED
        and linked to the best-matching failure log line.

        The threshold of 2 tokens prevents false positives caused by a
        single generic word accidentally appearing in both places.

      Step 2 — Pass confirmation:
        If no failure match is found, search PASSED/SUCCESS log lines for
        at least 1 token overlap.  If confirmed, mark PASSED with a note.

      Step 3 — Default pass:
        If neither a failure nor an explicit pass confirmation is found,
        the test case is assumed to have completed nominally → PASSED.

    Each test case is enriched with:
      status        — "PASSED" or "FAILED"
      matched_log   — the exact failure log line (if FAILED)
      match_score   — number of overlapping tokens (evidence strength)
      failure_type  — RPA failure category (e.g. "Timeout", "DataError")
      match_reason  — human-readable explanation of the decision
      log_level     — level of the matched log line (ERROR, FAILED, etc.)
    """
    # Pre-split log entries into failure and passed groups for efficiency
    failure_entries = [e for e in log_entries if e["is_failure"]]
    passed_entries  = [e for e in log_entries if e["level"] in ("PASSED", "SUCCESS")]

    for tc in test_cases:
        # Build the test case's meaningful token set
        tc_tokens = _meaningful_tokens(tc["name"] + " " + tc["description"])

        # If the test case has no meaningful tokens, default to PASSED
        if not tc_tokens:
            tc.update({
                "status":       "PASSED",
                "matched_log":  "",
                "match_score":  0,
                "failure_type": "",
                "match_reason": "No specific keywords — defaulted to PASSED",
                "log_level":    "",
            })
            continue

        # ── Step 1: Search for the best-matching failure log line ──────────
        best_fail           = None   # the failure entry with the highest overlap
        best_score          = 0      # highest token overlap seen so far
        best_overlap_tokens = set()  # the actual shared tokens (shown as evidence)

        for entry in failure_entries:
            log_tokens  = _meaningful_tokens(entry["raw"])
            overlap_set = tc_tokens & log_tokens    # intersection
            overlap     = len(overlap_set)

            # Require at least 2 overlapping tokens to avoid false positives
            if overlap >= 2 and overlap > best_score:
                best_score          = overlap
                best_fail           = entry
                best_overlap_tokens = overlap_set

        if best_fail:
            # A failure match found — mark the test case as FAILED
            ftype = _classify_failure_type(best_fail["raw"])
            tc.update({
                "status":       "FAILED",
                "matched_log":  best_fail["raw"][:160],   # truncate for display
                "match_score":  best_score,
                "failure_type": ftype,
                "log_level":    best_fail["level"],
                "match_reason": (
                    f"Matched failure log via keywords: "
                    f"{', '.join(sorted(best_overlap_tokens)[:5])}. "   # top 5 tokens
                    f"Failure type: {ftype}."
                ),
            })
            continue   # no need to check pass entries — already marked FAILED

        # ── Step 2: Look for an explicit PASSED log confirmation ────────────
        confirmed_pass = None
        for entry in passed_entries:
            log_tokens = _meaningful_tokens(entry["raw"])
            if len(tc_tokens & log_tokens) >= 1:   # 1 token enough for pass confirm
                confirmed_pass = entry
                break

        # ── Step 3: Mark as PASSED (confirmed or nominal) ──────────────────
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
    """
    Compute KPI metrics from the matched test cases and all log entries.

    Returns a dict with:
      total                  — number of test cases
      passed                 — test cases with status == "PASSED"
      failed                 — test cases with status == "FAILED"
      success_rate           — percentage of passed cases (rounded to 1dp)
      total_log_lines        — total number of parsed log lines
      error_log_count        — number of log lines flagged as failures
      impacted_transactions  — up to 10 unique failure log snippets
      failure_type_breakdown — {failure_type: count} dict for failed TCs
    """
    total  = len(test_cases)
    passed = sum(1 for tc in test_cases if tc.get("status") == "PASSED")
    failed = sum(1 for tc in test_cases if tc.get("status") == "FAILED")

    # Avoid division-by-zero when no test cases have been loaded
    success_rate = round((passed / total) * 100, 1) if total else 0

    # Collect unique failure log lines as "impacted transactions"
    error_logs    = [e for e in log_entries if e["is_failure"]]
    impacted_txns = list({e["raw"][:60] for e in error_logs})[:10]

    # Count how many failed test cases belong to each failure type
    # (used for the "Failure Types Detected" badge strip in the KPI panel)
    type_counts: dict = {}
    for tc in test_cases:
        if tc.get("status") == "FAILED" and tc.get("failure_type"):
            ft = tc["failure_type"]
            type_counts[ft] = type_counts.get(ft, 0) + 1

    return {
        "total":                  total,
        "passed":                 passed,
        "failed":                 failed,
        "success_rate":           success_rate,
        "total_log_lines":        len(log_entries),
        "error_log_count":        len(error_logs),
        "impacted_transactions":  impacted_txns,
        "failure_type_breakdown": type_counts,
    }


def _detect_recurring_issues(log_entries: list[dict]) -> list[dict]:
    """
    Identify failure messages that appear on multiple dates (recurring issues).

    Algorithm:
      1. For each failure log line, strip timestamps and replace all numbers
         with '#' to create a normalised message signature.
         Example: "SAP timeout at 09:04:00" → "SAP timeout at ##:##:##"
      2. Group these signatures by the dates they appeared on.
      3. Calculate the span (first occurrence to last occurrence in days).
      4. Flag as "persistent" if span >= 2 days OR seen on >= 3 separate dates.
         Persistent issues are highlighted as SLA risks in the AI summary.

    Returns a list of issue dicts sorted by: most occurrences → longest span.
    (Up to 20 issues returned to keep the UI manageable.)
    """
    # Map each normalised issue message to the set of dates it appeared on
    issue_dates: dict[str, set] = defaultdict(set)

    for entry in log_entries:
        if not entry["is_failure"]:
            continue

        # Strip timestamps (e.g. "2024-01-20 09:04:00") from the message
        normalised = re.sub(
            r"\d{4}[-/]\d{2}[-/]\d{2}[T \t]\d{2}:\d{2}(:\d{2})?", "", entry["raw"]
        )
        # Replace all remaining numbers with '#' so "line 142" and "line 99"
        # are treated as the same recurring issue
        normalised = re.sub(r"\b\d+\b", "#", normalised).strip()[:120]

        issue_dates[normalised].add(entry["date"])   # add the date of this occurrence

    recurring = []
    for msg, dates in issue_dates.items():
        sorted_dates = sorted(dates)   # ascending date order
        span_days = 0

        if len(sorted_dates) >= 2:
            # Calculate calendar span from earliest to latest occurrence
            try:
                d1 = datetime.strptime(sorted_dates[0],  "%Y-%m-%d")
                d2 = datetime.strptime(sorted_dates[-1], "%Y-%m-%d")
                span_days = (d2 - d1).days
            except ValueError:
                # If date parsing fails, use occurrence count as proxy for span
                span_days = len(sorted_dates)

        recurring.append({
            "issue":       msg,
            "dates":       sorted_dates,
            "occurrences": len(dates),
            "span_days":   span_days,
            # Persistent = seen on 2+ different days OR at least 3 occurrences
            "persistent":  span_days >= 2 or len(sorted_dates) >= 3,
        })

    # Sort by most frequent first, then by longest span
    recurring.sort(key=lambda x: (-x["occurrences"], -x["span_days"]))
    return recurring[:20]   # return at most 20 to keep UI clean


def _build_rca(failure_entries: list[dict], rag_vectors: list[dict],
               rag_docs: list[str]) -> list[dict]:
    """
    Generate a Root Cause Analysis (RCA) entry for each unique failure log line.

    Two-phase approach:
      Phase 1 — Knowledge Base matching:
        Score each failure line against all RCA_KNOWLEDGE_BASE rules by
        counting keyword hits.  The rule with the most hits wins.

      Phase 2 — RAG context enrichment:
        Query the RAG engine for similar past failure lines in the corpus.
        If a similar issue is found (score > 0.05), its excerpt is appended
        to the RCA text to give the analyst historical context.

    Deduplication:
        An MD5 hash of the first 80 characters of each failure line is used
        as a signature.  Duplicate lines produce only one RCA entry.

    Returns up to 15 RCA entries sorted by input order (most recent first,
    since failure_entries is already in log order).
    """
    seen: set[str] = set()   # tracks already-processed failure signatures
    rcas = []

    for entry in failure_entries:
        # Create a short signature to deduplicate near-identical failures
        sig = hashlib.md5(entry["raw"][:80].encode()).hexdigest()[:8]
        if sig in seen:
            continue   # skip duplicate / very similar failure lines
        seen.add(sig)

        line_lower = entry["raw"].lower()

        # ── Phase 1: Knowledge base rule matching ──────────────────────────
        best_rule = None
        best_hits = 0
        for rule in RCA_KNOWLEDGE_BASE:
            # Count how many keywords in this rule's pattern appear in the line
            hits = sum(1 for kw in rule["pattern"] if kw in line_lower)
            if hits > best_hits:
                best_hits = hits
                best_rule = rule

        # ── Phase 2: RAG — find similar historical failures ─────────────────
        similar_docs = _query_rag(entry["raw"], rag_vectors, rag_docs, top_k=2)
        # Filter out very low-similarity results (score ≤ 0.05 is essentially noise)
        similar_issues = [
            {"score": round(s, 3), "excerpt": d[:120]}
            for s, d in similar_docs
            if s > 0.05
        ]

        # Select RCA text and fix from the best-matching rule (or fallback)
        rca_text = best_rule["rca"] if best_rule else "Root cause not deterministic — review stack trace."
        fix_text = best_rule["fix"] if best_rule else "Consult bot developer, review logs manually."
        category = best_rule["category"] if best_rule else "Unknown"

        # Augment RCA with RAG context if a similar past issue was found
        if similar_issues:
            rca_text += (
                f" (RAG: similar issue found — '{similar_issues[0]['excerpt'][:80]}')"
            )

        # Confidence score: base 40% + 20% per rule keyword hit + 10% if RAG found something
        confidence = min(100, 40 + best_hits * 20 + (10 if similar_issues else 0))

        rcas.append({
            "failure":       entry["raw"][:140],   # truncated for display
            "date":          entry["date"],
            "category":      category,
            "rca":           rca_text,
            "fix":           fix_text,
            "similar_issues": similar_issues,
            "confidence":    confidence,
        })

    return rcas[:15]   # cap at 15 to keep the UI scrollable


# =============================================================================
# 7. RPA PROCEDURE FIX PLAYBOOK
# =============================================================================
#
# Maps each failure type (from _classify_failure_type) to an ordered list of
# 5 concrete, actionable steps an RPA developer should follow to resolve it.
# These steps are rendered in the AI Summary panel under "RPA Fix Playbook".

_RPA_FIX_PLAYBOOK = {

    # ── Timeout: the bot waited too long for a response ──────────────────
    "Timeout": [
        "1. Check network connectivity between the bot host and the target service.",
        "2. Increase the timeout threshold in the bot configuration (e.g. WaitTimeout, ElementTimeout).",
        "3. Add a retry loop with exponential back-off (e.g. 3 attempts, 5s / 15s / 30s delay).",
        "4. Verify the target application/service is healthy and not under load.",
        "5. Alert the infrastructure team if the service is persistently unreachable.",
    ],

    # ── AuthFailure: login or credential problem ─────────────────────────
    "AuthFailure": [
        "1. Rotate the bot credentials in the Credential Store / CyberArk / Secret Manager.",
        "2. Verify the bot service account is not locked or expired in Active Directory.",
        "3. Check SSO/OAuth token expiry — refresh tokens if required.",
        "4. Confirm IAM roles and permissions have not changed since last successful run.",
        "5. Re-run the bot with updated credentials and verify login succeeds.",
    ],

    # ── UIFailure: the bot could not find a UI element ───────────────────
    "UIFailure": [
        "1. Verify the application version/build has not changed (UI regression).",
        "2. Update the XPath/CSS selector in the bot workflow to match the new UI layout.",
        "3. Add a 'wait for element' step before interacting to handle slow page loads.",
        "4. Take a screenshot at the point of failure for visual comparison.",
        "5. Re-run in attended mode to manually validate the selector before deploying.",
    ],

    # ── DataError: null/wrong-type data caused the failure ───────────────
    "DataError": [
        "1. Add null/empty checks before passing data to downstream steps.",
        "2. Validate input data against the expected schema at the start of each transaction.",
        "3. Log the actual value received and compare against the expected data type.",
        "4. Implement an exception handler to move failed transactions to an error queue.",
        "5. Review upstream data source (DB, API, file) for missing or malformed records.",
    ],

    # ── DBFailure: database connection or query failure ───────────────────
    "DBFailure": [
        "1. Check the database connection string and credentials in the bot config.",
        "2. Verify the DB server is running and reachable from the bot host.",
        "3. Review the SQL query for syntax errors or schema changes.",
        "4. Check for connection pool exhaustion — increase pool size or add a retry.",
        "5. Review the DB server logs for deadlocks or maintenance windows.",
    ],

    # ── APIFailure: external REST/HTTP call failed ────────────────────────
    "APIFailure": [
        "1. Verify the API endpoint URL and port are correct and the service is live.",
        "2. Check the HTTP status code — 401/403 = auth, 404 = endpoint changed, 5xx = server error.",
        "3. Validate the request payload matches the API contract (headers, content-type, body).",
        "4. Implement back-off retry for transient 5xx errors (max 3 retries).",
        "5. Contact the API owner if the endpoint has moved or the contract has changed.",
    ],

    # ── FileSystemError: file not found or permission denied ─────────────
    "FileSystemError": [
        "1. Verify the file/folder path exists and is accessible from the bot host.",
        "2. Check that the bot service account has read/write permissions on the target path.",
        "3. Confirm disk space is available on the target drive.",
        "4. Check if the file is locked by another process.",
        "5. Add a pre-check step to validate file existence before the main workflow.",
    ],

    # ── ResourceExhaustion: out-of-memory or CPU saturation ──────────────
    "ResourceExhaustion": [
        "1. Increase the JVM heap or process memory allocation for the bot runtime.",
        "2. Close unused file handles, browser windows, and DB connections after each transaction.",
        "3. Reduce the batch size per run to lower peak memory usage.",
        "4. Schedule the bot during off-peak hours to reduce resource contention.",
        "5. Monitor host CPU/RAM with an alerting rule — escalate if consistently >80%.",
    ],

    # ── ProcessAbort: bot was killed or crashed unexpectedly ─────────────
    "ProcessAbort": [
        "1. Review the orchestrator logs for kill signals or conflicting scheduled jobs.",
        "2. Check for duplicate running instances — ensure only one instance runs at a time.",
        "3. Add a global exception handler to capture and log unhandled exceptions.",
        "4. Implement a bot health-check ping every N minutes to detect silent crashes.",
        "5. Escalate to the bot developer if a traceback/stack trace is present in the logs.",
    ],

    # ── RuntimeException: generic unclassified exception ─────────────────
    "RuntimeException": [
        "1. Capture the full stack trace from the log and identify the failing module and line.",
        "2. Reproduce the failure in a test environment with the same input data.",
        "3. Add try/except blocks around the failing code section with meaningful error logging.",
        "4. Validate all external dependencies (files, DB, APIs) are available before execution.",
        "5. Deploy the fix to a staging environment, run regression tests, then promote to production.",
    ],
}

# Fallback playbook used when no failure type was classified
_DEFAULT_PLAYBOOK = [
    "1. Review the full log file for the exact error message and stack trace.",
    "2. Isolate the failing step by running the bot in debug/attended mode.",
    "3. Check all external dependencies (services, files, credentials) are available.",
    "4. Add exception handling and retry logic around the failing step.",
    "5. Re-test and monitor for recurrence after the fix is deployed.",
]


# =============================================================================
# 8. AI SUMMARY GENERATOR
# =============================================================================

def _generate_ai_summary(kpis: dict, rcas: list[dict],
                         recurring: list[dict], log_entries: list[dict],
                         test_cases: list[dict] = None) -> dict:
    """
    Generate a structured, RPA-aware AI summary as a dict of named sections.

    This function produces actionable intelligence without any external API call.
    It combines KPI data, matched test cases, RCA results, and recurring issue
    detection into five clearly separated sections that the UI renders as a
    visual "playbook" page.

    Sections returned:
      health        — "Healthy" / "Degraded" / "Critical"
      success_rate  — numeric success percentage (mirrors kpis["success_rate"])
      overview      — one-paragraph executive summary (multi-line string)
      what_happened — list of dicts, one per failed test case with failure detail
      fix_steps     — list of dicts {failure_type, steps[]}, one per unique type
      actions       — priority-ordered list of action strings
      rca_summary   — list of strings from the top RCA categories

    The sections are independent so the UI can render each one in its own
    styled block without post-processing.
    """
    sr         = kpis["success_rate"]
    health     = "Healthy" if sr >= 90 else ("Degraded" if sr >= 60 else "Critical")
    failed_tcs = [tc for tc in (test_cases or []) if tc.get("status") == "FAILED"]
    persistent = [r  for r  in recurring if r["persistent"]]

    # ── Section 1: Executive Overview ─────────────────────────────────────
    # A concise 2–3 line paragraph showing overall health and key numbers.
    overview_lines = [
        f"Bot Health: {health}  |  Success Rate: {sr}%  |  "
        f"{kpis['passed']} Passed / {kpis['failed']} Failed out of {kpis['total']} test cases.",
        f"{kpis['error_log_count']} error log lines detected across "
        f"{kpis['total_log_lines']} total log entries.",
    ]
    if persistent:
        # Add a warning if any issue has been recurring for 2+ days
        overview_lines.append(
            f"WARNING: {len(persistent)} recurring issue(s) persisting for 2+ days — SLA risk."
        )

    # ── Section 2: What Happened (per failed test case) ───────────────────
    # A structured breakdown of each failure so the user can see EXACTLY
    # which test case failed, WHY it failed, and WHAT log line triggered it.
    what_happened = []
    for tc in failed_tcs[:8]:   # cap at 8 to keep the UI readable
        what_happened.append({
            "tc_id":        tc.get("id",           ""),
            "tc_name":      tc.get("name",          ""),
            "failure_type": tc.get("failure_type",  "UnknownFailure"),
            "match_reason": tc.get("match_reason",  ""),
            "matched_log":  tc.get("matched_log",   ""),
            "log_level":    tc.get("log_level",     "ERROR"),
        })

    # ── Section 3: RPA Procedure Fix Steps ────────────────────────────────
    # For each UNIQUE failure type seen in failed test cases, look up the
    # corresponding 5-step RPA resolution guide from _RPA_FIX_PLAYBOOK.
    # Up to 4 unique failure types are included to keep the playbook focused.
    seen_types: list[str] = []
    fix_steps = []

    for tc in failed_tcs:
        ft = tc.get("failure_type", "")
        if ft and ft not in seen_types:
            seen_types.append(ft)
            steps = _RPA_FIX_PLAYBOOK.get(ft, _DEFAULT_PLAYBOOK)
            fix_steps.append({"failure_type": ft, "steps": steps})
        if len(seen_types) >= 4:
            break   # limit to 4 playbooks to avoid information overload

    # If no test case failures were matched (log-only upload) fall back to RCA
    if not fix_steps and rcas:
        for rca in rcas[:3]:
            ft = rca.get("category", "")
            if ft and ft not in seen_types:
                seen_types.append(ft)
                steps = _RPA_FIX_PLAYBOOK.get(ft, _DEFAULT_PLAYBOOK)
                fix_steps.append({
                    "failure_type": ft,
                    "steps":        steps,
                    "rca_note":     rca.get("fix", ""),   # include the 1-liner from RCA KB
                })

    # ── Section 4: Recommended Actions ────────────────────────────────────
    # Priority-based action list — the severity of the recommended action
    # scales with the success rate.
    actions = []
    if sr == 100:
        actions.append("All test cases passed. Monitor next scheduled run.")
    elif sr >= 90:
        # Mostly healthy — minor fixes only
        actions.append("Bot is mostly stable. Fix highlighted failures before the next run.")
        actions.append("Review error logs and add retry logic for transient failures.")
    elif sr >= 60:
        # Degraded — P2 incident level
        actions.append("Bot performance is degraded. Prioritise remediation immediately.")
        actions.append("Pause non-critical bot schedules until failures are resolved.")
        actions.append("Raise a P2 incident ticket and assign to the RPA developer.")
    else:
        # Critical — P1 incident level
        actions.append("CRITICAL: Halt all production runs immediately.")
        actions.append("Escalate to RPA Lead and Infrastructure team — raise a P1 incident.")
        actions.append("Do not re-enable until root cause is confirmed fixed and re-tested.")

    if persistent:
        # Highlight the longest-running persistent issue
        p = persistent[0]
        actions.append(
            f"Persistent issue detected for {p['span_days']} days: "
            f"\"{p['issue'][:70]}\" — schedule a dedicated RCA session."
        )

    if kpis["impacted_transactions"]:
        # Remind the analyst to re-process transactions that failed
        actions.append(
            f"Re-process {len(kpis['impacted_transactions'])} impacted transaction(s) "
            "after the fix is deployed."
        )

    # ── Section 5: Top RCA Summary ────────────────────────────────────────
    # A concise list showing the most common failure categories and the top
    # 3 individual RCA entries for quick reference.
    rca_summary = []
    if rcas:
        top_cats = Counter(r["category"] for r in rcas).most_common(3)
        rca_summary.append(
            "Top failure categories: "
            + ", ".join(f"{cat} ({cnt})" for cat, cnt in top_cats) + "."
        )
        for rca in rcas[:3]:
            rca_summary.append(
                f"[{rca['category']}] {rca['failure'][:80]} -> {rca['rca']}"
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


# =============================================================================
# 9. MULTI-DATE LOG COMPARISON
# =============================================================================

def _compare_logs(log_entries_by_date: dict[str, list[dict]]) -> list[dict]:
    """
    Compare failure patterns between consecutive log dates.

    For each adjacent pair of dates (sorted ascending), computes:
      new_failures       — failures present in the later date but NOT the earlier
      resolved_failures  — failures present in the earlier date but NOT the later
      persisting_failures — failures present in BOTH dates (still unresolved)
      trend              — "Improving" / "Worsening" / "Stable"
                           based on whether more issues were resolved or introduced

    Returns a list of comparison dicts (one per date pair).
    Requires at least 2 dates in the dataset to produce any comparisons.
    """
    comparisons = []
    dates = sorted(log_entries_by_date.keys())   # ascending date order

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        # Use the first 80 characters of each failure line as a signature
        # (truncating avoids noise from trailing timestamps / line numbers)
        prev_failures = {
            e["raw"][:80] for e in log_entries_by_date[prev_date] if e["is_failure"]
        }
        curr_failures = {
            e["raw"][:80] for e in log_entries_by_date[curr_date] if e["is_failure"]
        }

        new_failures  = curr_failures - prev_failures   # appeared in curr, not prev
        resolved      = prev_failures - curr_failures   # disappeared between dates
        persisting    = prev_failures & curr_failures   # still present in both

        # Determine the overall trend for this date transition
        if len(resolved) > len(new_failures):
            trend = "Improving"    # more issues fixed than introduced
        elif len(new_failures) > len(resolved):
            trend = "Worsening"    # more new issues than fixes
        else:
            trend = "Stable"       # equal new and resolved (could be 0/0 or n/n)

        comparisons.append({
            "from_date":           prev_date,
            "to_date":             curr_date,
            "new_failures":        list(new_failures)[:5],    # cap for display
            "resolved_failures":   list(resolved)[:5],
            "persisting_failures": list(persisting)[:5],
            "trend":               trend,
        })

    return comparisons


# =============================================================================
# 10. FLASK ROUTES (HTTP API)
# =============================================================================

def _ingest_logs(file_pairs: list[tuple[str, str]]) -> int:
    """
    Helper: parse a batch of (filename, text) pairs and append to _store.

    After parsing, rebuilds the RAG vector index from ALL failure lines
    currently in the store (including any previously uploaded files).
    This allows multiple upload calls to accumulate into a single corpus.

    Returns the number of NEW log lines parsed in this call.
    """
    all_entries = []
    for fname, text in file_pairs:
        entries = _parse_log_lines(text, fname)
        all_entries.extend(entries)

    # Append to the running store (supports multi-file / incremental upload)
    _store["log_entries"].extend(all_entries)

    # Rebuild the RAG index over ALL failure lines (including previous uploads)
    failure_docs = [e["raw"] for e in _store["log_entries"] if e["is_failure"]]
    if failure_docs:
        _store["rag_vectors"], _ = _tf_idf_vectors(failure_docs)
        _store["rag_docs"]       = failure_docs
    else:
        # No failures yet — reset vectors to empty
        _store["rag_vectors"], _store["rag_docs"] = [], []

    return len(all_entries)   # count of lines added in this call only


@app.route("/")
def index():
    """Serve the main dashboard HTML page."""
    return render_template("index.html")


@app.route("/upload_logs", methods=["POST"])
def upload_logs():
    """
    Accept one or more log files uploaded via the browser file-picker.

    Expects a multipart/form-data POST with field name "logs".
    Multiple files can be attached in a single request.

    On success returns JSON:
        { message, files: [filename, ...], total_entries }

    On failure returns JSON with "error" key and HTTP 400.
    """
    files = request.files.getlist("logs")   # get all files with name="logs"

    # Validate that at least one file with a non-empty filename was provided
    if not files or not any(f.filename for f in files):
        return jsonify({
            "error": "No log files received. Please select at least one .log or .txt file."
        }), 400

    # Read and decode each file; use secure_filename to sanitise the name
    pairs = []
    for f in files:
        fname = secure_filename(f.filename) or f.filename   # fallback to original if sanitised empty
        text  = f.read().decode("utf-8", errors="replace")  # replace undecodable bytes
        pairs.append((fname, text))

    count = _ingest_logs(pairs)

    return jsonify({
        "message":       f"Parsed {count} log lines from {len(pairs)} file(s).",
        "files":         [p[0] for p in pairs],
        "total_entries": len(_store["log_entries"]),
    })


@app.route("/upload_test_cases", methods=["POST"])
def upload_test_cases():
    """
    Accept a test case CSV/TXT file uploaded via the browser file-picker.

    Expects a multipart/form-data POST with field name "test_cases".
    Replaces any previously uploaded test cases in the store.

    CSV format:  TestID, Name, Description, Expected
    Example:     TC-001, SAP Login, Verify login succeeds, PASS

    On success returns JSON: { message, test_cases: [...] }
    On failure returns JSON with "error" key and HTTP 400.
    """
    f = request.files.get("test_cases")   # single file expected
    if not f or not f.filename:
        return jsonify({"error": "No test case file received."}), 400

    text  = f.read().decode("utf-8", errors="replace")
    cases = _parse_test_cases(text)

    # Replace (not append) test cases so a fresh upload starts clean
    _store["test_cases"] = cases

    return jsonify({
        "message":    f"Loaded {len(cases)} test case(s).",
        "test_cases": cases,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Run the full analysis pipeline on the currently stored data and return
    a comprehensive JSON payload used to populate all dashboard panels.

    Pipeline:
      1. If test cases AND logs are available → match test cases to logs.
      2. If only logs are available → auto-generate synthetic test cases
         from the failure entries so the KPI panel always has data.
      3. Compute KPIs (total / passed / failed / success rate).
      4. Build RCA entries for each unique failure.
      5. Detect recurring/persistent issues across dates.
      6. Compare failures date-by-date (if multiple log dates present).
      7. Generate the AI summary dict.

    Returns JSON with keys:
        kpis, test_cases, rcas, recurring_issues,
        log_comparisons, ai_summary, dates_covered
    """
    log_entries = _store["log_entries"]
    test_cases  = _store["test_cases"]

    # Refuse to analyse if nothing has been uploaded yet
    if not log_entries and not test_cases:
        return jsonify({"error": "No data found. Please upload log files first."}), 400

    # ── Step 1 / 2: Resolve test cases ────────────────────────────────────
    if test_cases and log_entries:
        # Both present — match user-provided test cases against log entries
        test_cases = _match_test_cases_to_logs(test_cases, log_entries)
        _store["test_cases"] = test_cases

    elif not test_cases:
        # No test case file uploaded — auto-generate from log failures
        failure_entries = [e for e in log_entries if e["is_failure"]]
        test_cases = []

        # Create one FAILED test case per unique failure log line (up to 20)
        for i, fe in enumerate(failure_entries[:20], 1):
            test_cases.append({
                "id":          f"AUTO-{i:03d}",
                "name":        fe["raw"][:50],
                "description": fe["raw"][:80],
                "expected":    "PASS",
                "status":      "FAILED",
                "matched_log": fe["raw"][:120],
                "match_reason": "Auto-generated from failure log line.",
                "failure_type": _classify_failure_type(fe["raw"]),
                "log_level":   fe["level"],
                "match_score": 0,
            })

        # Also create a few PASSED entries to represent nominal executions
        pass_count = max(1, len(log_entries) - len(failure_entries))
        for i in range(min(pass_count, 5)):
            test_cases.append({
                "id":          f"AUTO-P{i+1:03d}",
                "name":        f"Nominal Execution #{i+1}",
                "description": "No failure detected in corresponding log section",
                "expected":    "PASS",
                "status":      "PASSED",
                "matched_log": "",
                "match_reason": "No failure detected in logs.",
                "failure_type": "",
                "log_level":   "INFO",
                "match_score": 0,
            })

        _store["test_cases"] = test_cases

    # ── Steps 3–7: Run all analysis functions ─────────────────────────────
    kpis            = _compute_kpis(test_cases, log_entries)
    failure_entries = [e for e in log_entries if e["is_failure"]]
    rcas            = _build_rca(failure_entries, _store["rag_vectors"], _store["rag_docs"])
    recurring       = _detect_recurring_issues(log_entries)

    # Group log entries by date for multi-date comparison
    by_date: dict[str, list] = defaultdict(list)
    for e in log_entries:
        by_date[e["date"]].append(e)

    # Only compare if there are at least 2 distinct dates in the dataset
    comparisons = _compare_logs(dict(by_date)) if len(by_date) > 1 else []

    ai_summary = _generate_ai_summary(kpis, rcas, recurring, log_entries, test_cases)

    return jsonify({
        "kpis":              kpis,
        "test_cases":        test_cases,
        "rcas":              rcas,
        "recurring_issues":  recurring,
        "log_comparisons":   comparisons,
        "ai_summary":        ai_summary,
        "dates_covered":     sorted(by_date.keys()),
    })


@app.route("/reset", methods=["POST"])
def reset():
    """
    Clear all in-memory data from the store, effectively starting a fresh session.

    Called by the ↺ Reset button in the UI.  The response includes the post-clear
    counts so the JavaScript can verify the backend was actually cleared.

    Returns JSON: { message, entries: 0, test_cases: 0 }
    """
    _store["log_entries"].clear()   # clear parsed log lines
    _store["test_cases"].clear()    # clear test case definitions
    _store["rag_vectors"].clear()   # clear TF-IDF vectors
    _store["rag_docs"].clear()      # clear RAG corpus strings

    return jsonify({
        "message":    "Session reset.",
        "entries":    len(_store["log_entries"]),    # should be 0
        "test_cases": len(_store["test_cases"]),     # should be 0
    })


@app.route("/status", methods=["GET"])
def status():
    """
    Health-check endpoint that returns the current store counts.

    Useful for debugging (e.g. open in browser to confirm reset worked)
    and for automated tests to verify state without triggering analysis.

    Returns JSON: { log_entries, test_cases, rag_docs }
    """
    return jsonify({
        "log_entries": len(_store["log_entries"]),
        "test_cases":  len(_store["test_cases"]),
        "rag_docs":    len(_store["rag_docs"]),
    })


# =============================================================================
# 11. ENTRY POINT
# =============================================================================
#
# IMPORTANT — Why these exact flags:
#
#   host="0.0.0.0"      — Bind to ALL interfaces (not just loopback).
#                          Fixes "network error" when browser sends request to
#                          127.0.0.1 but Flask is only listening on ::1 (IPv6).
#
#   port=5000           — Standard dev port; change to 5001 if 5000 is taken.
#
#   debug=False         — Completely disables the Werkzeug reloader & debugger.
#                          On Flask 3.x + Windows, debug=True with use_reloader=False
#                          can STILL spawn a child process in some Python builds,
#                          causing the child to have an empty _store dict and
#                          returning blank / error responses to the browser.
#                          Setting debug=False is the safest option for production-like runs.
#
#   threaded=True       — Allow Flask to handle concurrent requests (browser
#                          may fire multiple fetches in parallel — upload + analyze).
#                          Without this, the second request blocks until the first finishes,
#                          which can look like a network hang.
#
#   use_reloader=False  — Belt-and-suspenders: explicitly disable reloader even
#                          when debug=False (some Werkzeug versions ignore debug flag).

if __name__ == "__main__":
    import socket

    # ── Port availability check ────────────────────────────────────────────
    # Give a clear error if port 5000 is already in use (e.g. stale old process)
    # rather than silently failing with "address in use".
    port = 5000
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0", port))
        probe.close()
    except OSError:
        print(f"\n[ERROR] Port {port} is already in use!")
        print("        Kill the other process first, or change port= to 5001.\n")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("  RPA Bot Monitoring Dashboard")
    print(f"  Running at: http://127.0.0.1:{port}")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    app.run(
        host="0.0.0.0",          # bind all interfaces — fixes IPv6/IPv4 mismatch
        port=port,
        debug=False,             # NO reloader, NO debugger — single stable process
        threaded=True,           # handle concurrent fetch() calls from browser
        use_reloader=False,      # belt-and-suspenders: reloader explicitly off
    )
