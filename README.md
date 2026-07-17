# RPA Bot Monitoring Dashboard

A self-contained Python Flask web application for monitoring RPA (Robotic Process Automation) bots.  
**No external API key required** — uses a local TF-IDF RAG engine for AI-powered analysis.

---

## Features

| Feature | Description |
|---|---|
| **Upload Execution Logs** | Upload `.log` / `.txt` files from multiple dates |
| **Upload Test Cases** | CSV format: `TestID, Name, Description, Expected` |
| **KPI Dashboard** | Total / Passed / Failed / Success Rate with donut chart |
| **Failure Detection** | Automatic regex + keyword-based failure identification |
| **RCA Generation** | Pattern-matched root cause + recommended fix per failure |
| **Local RAG Engine** | TF-IDF cosine similarity — finds similar past issues with zero API calls |
| **Multi-Day Comparison** | Detects new / resolved / persisting failures across log dates |
| **Recurring Issue Detection** | Flags issues persisting 2–7+ days across log files |
| **Impacted Transactions** | Lists affected transaction IDs from error logs |
| **AI Summary** | Structured natural-language analysis: health status, top issues, action plan |
| **Raw Log Viewer** | Colour-coded log viewer with level filters |

---

## Quick Start

```bash
# 1. Install dependencies
pip install flask werkzeug

# 2. Run the server
python app.py

# 3. Open browser
http://localhost:5000
```

---

## Usage

### Option A — Sample Data (Fastest)
1. Open `http://localhost:5000`
2. Click **✨ Load Sample Data** — loads 4 log files spanning 7 days + 12 test cases
3. Click **Upload Files**
4. Click **Run Analysis**
5. Navigate panels in the left sidebar

### Option B — Your Own Files

**Log file format (`.log` / `.txt`):**
```
2024-01-20 07:00:15 ERROR Connection timeout: SAP endpoint did not respond
2024-01-20 07:01:00 PASSED Test: LoadConfiguration - Config loaded
2024-01-20 07:03:00 FAILED Transaction TXID-001 failed: Amount mismatch
```

**Multi-date detection** — embed the date in the filename:
```
2024-01-14_mybot.log
2024-01-16_mybot.log   ← dates auto-extracted from filename
2024-01-20_mybot.log
```
Or include ISO dates (`YYYY-MM-DD`) directly in each log line.

**Test case file (`.csv`):**
```
TC-001,SAP Connection Test,Bot connects to SAP ERP,PASS
TC-002,Validate Invoice Total,Check invoice amount,PASS
```

---

## Architecture

```
app.py
 ├── _tokenize / _tf_idf_vectors / _cosine_sim / _query_rag  ← Local RAG Engine
 ├── _parse_log_lines           ← Log parser (level, date, failure detection)
 ├── _parse_test_cases          ← CSV test case parser
 ├── _match_test_cases_to_logs  ← Heuristic test case → log linking
 ├── _compute_kpis              ← KPI computation
 ├── _detect_recurring_issues   ← Multi-day recurrence detection
 ├── _build_rca                 ← Rule-based + RAG root cause analysis
 ├── _compare_logs              ← Date-by-date failure diff
 └── _generate_ai_summary       ← Template-driven AI summary

templates/index.html            ← Full SPA (HTML + CSS + Vanilla JS)

sample_data/
 ├── test_cases.csv
 ├── 2024-01-14_invoice_bot.log
 ├── 2024-01-16_invoice_bot.log
 ├── 2024-01-18_invoice_bot.log
 └── 2024-01-20_invoice_bot.log
```

---

## Supported Log Levels

| Level | Colour |
|---|---|
| `ERROR` / `FAILED` / `CRITICAL` | 🔴 Red |
| `WARNING` | 🟡 Yellow |
| `PASSED` / `SUCCESS` | 🟢 Green |
| `INFO` | ⚪ Grey |

---

## RCA Knowledge Base Categories

`Network` · `Auth` · `UI` · `Database` · `FileSystem` · `API` · `Resource` · `DataValidation` · `Process` · `License`

---

## Requirements

- Python 3.10+
- Flask ≥ 3.0
- Werkzeug ≥ 3.0
- No other dependencies (RAG engine is pure Python stdlib)
