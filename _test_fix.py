import sys
sys.path.insert(0, '.')
import app as a

log_text = """2024-07-17 09:00:00 INFO  Bot started
2024-07-17 09:01:00 PASSED Test: LoadConfiguration - Config loaded successfully
2024-07-17 09:02:00 PASSED Test: ConnectToDatabase - DB connection established
2024-07-17 09:03:00 PASSED Test: CheckSupplierCode - Supplier ABC123 validated
2024-07-17 09:04:00 ERROR Connection timeout: SAP endpoint not responding within 30s
2024-07-17 09:05:00 FAILED Transaction TXID-001 failed: Amount mismatch 1200 != 1500
2024-07-17 09:06:00 ERROR Null pointer exception in InvoiceParser module line 142
"""

tc_text = """TC-001,SAP Connection Test,Check SAP endpoint connection,PASS
TC-002,Load Configuration,Load bot config file,PASS
TC-003,Database Connectivity,Connect to DB2 database,PASS
TC-004,Process Invoice TXID-001,Process transaction TXID-001 amount,PASS
TC-005,Validate Invoice Parser,Run InvoiceParser module validation,PASS
"""

entries = a._parse_log_lines(log_text, '2024-07-17_bot.log')
cases   = a._parse_test_cases(tc_text)
matched = a._match_test_cases_to_logs(cases, entries)

print("Test Case Results:")
for tc in matched:
    print(f"  {tc['id']} | {tc['name'][:40]:40s} | {tc['status']}")

passed = sum(1 for tc in matched if tc['status'] == 'PASSED')
failed = sum(1 for tc in matched if tc['status'] == 'FAILED')
print(f"\nTotal={len(matched)}  Passed={passed}  Failed={failed}  Rate={round(passed/len(matched)*100,1)}%")

assert failed == 2, f"Expected 2 failures, got {failed}"
assert passed == 3, f"Expected 3 passes, got {passed}"
print("PASS: Correct 3 passed / 2 failed split confirmed")
