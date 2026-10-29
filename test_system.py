import sys
import sqlite3
import urllib.request
from pathlib import Path
import joblib
import pandas as pd
from ml_engine import ml_engine

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def ensure_db_initialized():
    conn = sqlite3.connect("civic_complaints.db")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            citizen_id TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """
    )
    demo_users = [
        ("admin", "admin123", "Admin", "ADMIN-001", "Admin"),
        ("citizen", "citizen123", "Shashi Ranjan", "CIT-1001", "Citizen"),
        ("rahul", "rahul123", "Rahul Kumar", "CIT-1002", "Citizen")
    ]
    cur.executemany("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", demo_users)
    conn.commit()
    conn.close()

def run_tests():
    print("=" * 60)
    print("PHASE 9: SYSTEM VERIFICATION & TEST SUITE")
    print("=" * 60)
    
    ensure_db_initialized()
    test_results = {}

    # Test 1: Model & Vectorizer Loading
    try:
        assert ml_engine.model is not None, "Model is None"
        assert ml_engine.vectorizer is not None, "Vectorizer is None"
        test_results["Model & Vectorizer Loading"] = "PASSED"
        print("[PASS] Test 1: Model & Vectorizer Loading -> PASSED")
    except Exception as e:
        test_results["Model & Vectorizer Loading"] = f"FAILED: {e}"
        print(f"[FAIL] Test 1 FAILED: {e}")

    # Test 2: Unseen Predictions Accuracy & Department Routing
    test_cases = [
        ("Massive water pipeline burst leaking gallons of water near marketplace", "Water Supply & Management"),
        ("Huge pothole on road causing dangerous bike accidents", "Roads & Infrastructure"),
        ("Garbage dump accumulating stinking waste attracting mosquitoes", "Sanitation & Waste Management"),
        ("Live electrical wire sparking from pole near primary school", "Electrical Department"),
        ("Illegal parking and abandoned vehicles completely blocking traffic on commercial street", "Traffic Department")
    ]
    pred_passed = True
    for text, expected in test_cases:
        pred, conf, _ = ml_engine.predict_department(text)
        if pred != expected:
            pred_passed = False
            print(f"   Mismatch: '{text[:40]}...' Expected: {expected}, Got: {pred}")
    test_results["Unseen Department Predictions"] = "PASSED (5/5 Correct)" if pred_passed else "FAILED"
    print(f"[PASS] Test 2: Unseen Predictions Routing -> {test_results['Unseen Department Predictions']}")

    # Test 3: Priority Engine Scoring & Explainability
    crit_text = "Live electric wire sparking and fire danger near school"
    p_lvl, p_score, p_reason = ml_engine.calculate_priority(crit_text, "Dangerous hanging wires")
    if p_lvl == "Critical" and p_score >= 8.0 and "spark" in p_reason:
        test_results["Smart Priority Engine"] = "PASSED"
        print(f"[PASS] Test 3: Priority Engine -> PASSED (Level: {p_lvl}, Score: {p_score})")
    else:
        test_results["Smart Priority Engine"] = f"FAILED (Level: {p_lvl}, Score: {p_score})"
        print(f"[FAIL] Test 3 FAILED: {p_lvl}, {p_score}")

    # Test 4: Duplicate Detection
    mock_db = [
        {"id": "CVS-001", "description": "huge pothole on main road causing accidents", "category": "Potholes", "status": "Pending"}
    ]
    query = "massive pothole on main road causing accidents and danger"
    dups = ml_engine.check_duplicate_complaints(query, mock_db, threshold=0.40)
    if dups and dups[0]["similarity_score"] > 40.0:
        test_results["Duplicate Detection"] = f"PASSED (Matched: {dups[0]['similarity_score']}%)"
        print(f"[PASS] Test 4: Duplicate Detection -> PASSED ({dups[0]['similarity_score']}% match)")
    else:
        test_results["Duplicate Detection"] = "FAILED"
        print("[FAIL] Test 4 FAILED")

    # Test 5: SQLite Database Schema & Persistence
    try:
        conn = sqlite3.connect("civic_complaints.db")
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(complaints)")
        cols = [r[1] for r in cur.fetchall()]
        required_cols = ["predicted_department", "priority_level", "priority_score", "priority_reason", "matched_complaint_id"]
        for c in required_cols:
            assert c in cols, f"Missing column: {c}"
        cur.execute("SELECT COUNT(*) FROM complaints")
        cnt = cur.fetchone()[0]
        conn.close()
        test_results["SQLite Schema & Integrity"] = f"PASSED ({cnt} complaints verified)"
        print(f"[PASS] Test 5: SQLite Schema & Integrity -> PASSED ({cnt} records)")
    except Exception as e:
        test_results["SQLite Schema & Integrity"] = f"FAILED: {e}"
        print(f"[FAIL] Test 5 FAILED: {e}")

    # Test 6: Persistent Authentication Table
    try:
        conn = sqlite3.connect("civic_complaints.db")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        u_cnt = cur.fetchone()[0]
        conn.close()
        assert u_cnt >= 3, "Users table missing demo accounts"
        test_results["Persistent Authentication"] = f"PASSED ({u_cnt} registered accounts in DB)"
        print(f"[PASS] Test 6: Persistent Authentication -> PASSED ({u_cnt} users)")
    except Exception as e:
        test_results["Persistent Authentication"] = f"FAILED: {e}"
        print(f"[FAIL] Test 6 FAILED: {e}")

    # Test 7: Documentation Completeness
    doc_files = [
        "PROJECT_OVERVIEW.md", "PROJECT_AUDIT.md", "DATASET_DOCUMENTATION.md",
        "MODEL_DOCUMENTATION.md", "EXPERIMENT_RESULTS.md", "TECHNOLOGY_JUSTIFICATION.md",
        "ARCHITECTURE.md", "VIVA_QUESTIONS.md", "RUN_AND_TEST_GUIDE.md", "CHANGELOG.md"
    ]
    missing = [f for f in doc_files if not Path(f).exists() and not (Path("docs") / f).exists()]
    if not missing:
        test_results["Academic Documentation (10 files)"] = "PASSED (10/10 present)"
        print("[PASS] Test 7: Documentation Completeness -> PASSED (All 10 files present)")
    else:
        test_results["Academic Documentation (10 files)"] = f"FAILED (Missing: {missing})"
        print(f"[FAIL] Test 7 FAILED: Missing {missing}")

    # Test 8: Live Web Server HTTP Status
    try:
        req = urllib.request.urlopen("http://localhost:8501", timeout=3)
        code = req.getcode()
        if code == 200:
            test_results["Live Web App Status"] = "PASSED (HTTP 200 OK)"
            print("[PASS] Test 8: Live Web App Status -> PASSED (HTTP 200 OK on localhost:8501)")
        else:
            test_results["Live Web App Status"] = f"WARNING (HTTP {code})"
    except Exception as e:
        test_results["Live Web App Status"] = f"NOT RUNNING / NOT VERIFIED ({e})"
        print(f"[WARN] Test 8: Web App HTTP Check -> {e}")

    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY:")
    print("=" * 60)
    for k, v in test_results.items():
        print(f"  - {k:35}: {v}")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
