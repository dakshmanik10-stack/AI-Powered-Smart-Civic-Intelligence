# PROJECT AUDIT REPORT
**Project Name:** AI-Powered Civic Complaint Intelligence System  
**Audit Date:** October 29, 2026  
**Auditor:** Automated Engineering & AI Audit  
**Codebase Location:** `Civic-Complaint/`  

---

## 1. Executive Summary

This audit provides an authentic, evidence-based evaluation of the Civic Complaint Management project prior to any codebase modifications. The audit examined all source code, machine learning artifacts, datasets, database structures, and runtime behaviors.

**Key Findings:**
1. **Frontend & Basic CRUD:** The Streamlit application (`app.py`) is functional with citizen login/signup, complaint submission with attachment support, local SQLite database persistence, and an admin view for updating complaint statuses.
2. **AI Model Disconnection:** While a Logistic Regression model (`civic_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`) were trained using `main.py`, **`app.py` has zero imports or calls to `joblib`, the model, or the vectorizer**. Complaint categories are selected manually by users via UI pills (`st.pills`).
3. **Severe Train/Test Data Leakage:** In the training dataset (`municipal_training_set_1100.csv`), out of 1,100 rows, **653 rows have duplicate issue descriptions** (only 447 unique descriptions exist). Because `main.py` performed an unstratified random 80/20 train/test split without deduplication, identical complaint sentences exist in both training and test sets, artificially inflating the reported accuracy of ~88%.
4. **Mock Admin Dashboard Data:** The Admin GIS Dashboard contains a "Total Complaints" trend line chart rendered using **hardcoded mock numbers from February 2023** rather than actual database records.
5. **Priority Mechanism:** Completely absent in `app.py`. No priority score, severity score, or priority level is calculated or displayed to citizens or administrators.
6. **Model Comparison:** Not performed. Only a single Logistic Regression classifier was trained.
7. **Session Ephemerality:** Dynamically registered citizen accounts are stored only in memory (`st.session_state`), meaning any server restart or page refresh erases them, while complaints alone persist in SQLite.

---

## 2. File-by-File Inventory & Implementation Status

| File / Artifact | Stated Purpose | Actual Implementation Status | Issues / Gaps Identified |
| :--- | :--- | :--- | :--- |
| `app.py` (797 lines) | Streamlit User & Admin Interface | **Partially Working**<br>Citizen submission, DB saving, Admin status updates work. | No ML integration; hardcoded trend chart; ephemeral user sessions; placeholder upvotes/downvotes. |
| `main.py` (78 lines) | ML Model Training Pipeline | **Working (Standalone)**<br>Trains TF-IDF + Logistic Regression; saves `.pkl`. | Severe data leakage due to 653 duplicate texts; no model comparison; unstratified split on rare classes; label fragmentation. |
| `cluster_complaints.py` (62 lines) | Unsupervised Clustering | **Experimental Script**<br>Tests Sentence-Transformers (`all-MiniLM-L6-v2`) + KMeans(10). | Evaluates only first 200 rows; not connected to `app.py`; requires PyTorch/sentence-transformers which are resource-heavy. |
| `inspect_data.py` (18 lines) | Dataset inspection utility | **Working (Utility)** | Simple print statements for summary and head. |
| `EDA_and_Preprocessing.ipynb` (1 code cell) | Exploratory Data Analysis | **Incomplete Draft** | Contains only 1 code cell plotting top 5 departments and a simple regex function. No complete EDA. |
| `civic_complaints.db` (SQLite) | Relational Database | **Working**<br>Table `complaints` with 11 rows. | Only `complaints` table exists; no `users` table; photos/videos stored as raw binary blobs; no priority columns. |
| `civic_model.pkl` (127 KB) | Serialized Classifier | **Trained**<br>`LogisticRegression(max_iter=1000)` | Trained on 23 fragmented classes with leakage; uncalibrated probabilities. |
| `tfidf_vectorizer.pkl` (15 KB) | Serialized Vectorizer | **Trained**<br>TF-IDF with 683 vocabulary features | Fit on duplicate-contaminated data. |
| `municipal_training_set_1100.csv` | Training Dataset (1,100 rows, 26 cols) | **Present**<br>Benchmark synthetic municipal dataset | 653 duplicate descriptions; unstandardized department names; US-centric cities (NYC, Chicago, Phoenix). |
| `requirements.txt` | Environment specification | **Present** | Created during initial inspection; missing pinned version bounds. |

---

## 3. Detailed Machine Learning Audit

### 3.1 What Models Have Actually Been Trained?
- **Trained Model:** `sklearn.linear_model.LogisticRegression(max_iter=1000)`
- **Vectorizer:** `sklearn.feature_extraction.text.TfidfVectorizer()` (vocabulary size = 683 features)
- **Trained Classes (23 classes):**
  `['Code Enforcement', 'Community Affairs', 'Community Services', 'Drainage Authority', 'Electrical', 'Electrical Department', 'Electrical Services', 'Event Management', 'Noise Control', 'Parks & Recreation', 'Parks Department', 'Public Safety', 'Public Works', 'Road Maintenance', 'Roads', 'Roads & Highways', 'Sanitation', 'Traffic Department', 'Waste Management', 'Water Department', 'Water Management', 'Water Services', 'Waterworks']`

### 3.2 Has Any Model Comparison Been Performed?
- **Finding: NO.**
- Only `LogisticRegression` was executed. No Naive Bayes, Linear SVM, Decision Trees, or Random Forests were trained or benchmarked. No `model_comparison.csv` exists.

### 3.3 Does `app.py` Load and Use the Trained Model?
- **Finding: ABSOLUTELY NOT.**
- Verification:
  - `import joblib`: **False**
  - `joblib.load`: **False**
  - `civic_model.pkl` reference: **False**
  - `tfidf_vectorizer.pkl` reference: **False**
  - Model prediction function: **False**
- Citizen complaint submission in `app.py` line 390 relies strictly on `st.pills("Category *", CATEGORIES)` where the citizen manually picks one of 15 predefined category names.

### 3.4 Is Priority ML-Based, Rule-Based, or Manual?
- **Finding: COMPLETELY ABSENT in `app.py`.**
- Although the CSV dataset contains `priority_level` (`medium: 405`, `high: 273`, `critical: 215`, `low: 207`), `priority_score`, and `severity_score`, neither the training script nor `app.py` utilizes these fields. The word "priority" does not appear in `app.py`.

### 3.5 Duplicate Detection Status
- **Finding: NOT IMPLEMENTED in `app.py`.**
- In `cluster_complaints.py`, an experimental KMeans clustering script exists for 200 samples, but no duplicate complaint detection or similarity matching exists in the citizen portal or admin dashboard.

---

## 4. Dataset & Data Quality Audit

### 4.1 Schema & Columns
- **Total Rows:** 1,100
- **Total Columns:** 26
- **Key Columns:** `report_id`, `issue_type`, `issue_description`, `municipal_department`, `priority_level`, `priority_score`, `severity_score`, `area_importance`, `citizen_reports_count`, `location_city`, `latitude`, `longitude`.

### 4.2 Critical Data Quality Issues
1. **Severe Duplicate Contamination (Data Leakage):**
   - Out of 1,100 rows, there are only **447 unique issue descriptions**.
   - **653 rows are exact duplicate descriptions** paired with slightly modified resident IDs or locations.
   - In `main.py`, splitting without deduplicating caused identical text to appear in both training and test sets. This creates artificial memorization rather than generalization.
2. **Class Fragmentation / Label Inconsistency:**
   - Notice duplicate departmental jurisdictions in labels:
     - `Water Department` (153 rows), `Water Management` (5), `Water Services` (5), `Waterworks` (5).
     - `Roads & Highways` (129 rows), `Road Maintenance` (9), `Roads` (6).
     - `Electrical Department` (115 rows), `Electrical Services` (20), `Electrical` (6).
     - `Sanitation` (177 rows), `Waste Management` (5).
     - `Parks & Recreation` (7 rows), `Parks Department` (2).
   - This fragmentation splits the training data into tiny micro-classes (some with only 2 or 3 samples), damaging macro-F1 score (which was only 0.45 in `main.py`).
3. **Missing Values:**
   - `municipal_department`: 7 missing rows.
   - `resident_name`: 63 missing rows.
   - `resolution_datetime` & `resolution_notes`: 830 missing rows.
4. **Geographic Distribution:**
   - Locations are exclusively US cities (`New York`: 212, `Chicago`: 211, `Houston`: 203, `Phoenix`: 191, `Los Angeles`: 149).

---

## 5. User Interface & Dashboard Data Source Audit

| Dashboard Element | Location in `app.py` | Data Source | Verification |
| :--- | :--- | :--- | :--- |
| Citizen Metric Cards (Pending, Progress, Resolved) | Lines 468-472 | `st.session_state.reports` filtered by `citizen_id` | **Authentic** (Reflects DB) |
| Citizen Complaint Cards | Lines 480-518 | `st.session_state.reports` | **Authentic** (Reflects DB) |
| Upvote / Downvote Buttons | Lines 507-516 | Static button with `# demo placeholder` | **Incomplete Placeholder** (No DB write) |
| Admin Recent Complaints Table | Lines 526-573 | `st.session_state.reports` | **Authentic** (Reflects DB) |
| Admin Status Updates | Lines 574-583 | `update_report_statuses()` to SQLite | **Authentic** (Persists to DB) |
| Admin Metrics (Pending, Progress, Resolved) | Lines 596-598 | `st.session_state.reports` | **Authentic** (Reflects DB) |
| Admin Service Level Gauge | Lines 602-622 | Calculated: `resolved / total * 100` | **Authentic** (Reflects DB) |
| Admin Feedback Rating | Lines 626-630 | Hardcoded text `⭐ 4.6 / 5` | **Hardcoded Mock** |
| Admin "Total Complaints" Trend Line | Lines 634-644 | Hardcoded array `[140, 240, 210, ...]` from Feb 2023 | **Hardcoded Mock** |
| Admin "Complaints in Category" Bar Chart | Lines 667-705 | Aggregated from `report["category"]` | **Authentic** (Reflects DB) |

---

## 6. Security, Architecture & Reliability Audit

1. **Authentication Vulnerabilities:**
   - Admin credentials are hardcoded in source code (`admin` / `admin123`).
   - Plaintext password verification with no password hashing (bcrypt/Argon2).
   - Citizen signups live only in `st.session_state`. Restarting Streamlit deletes all created accounts.
2. **Database Limitations:**
   - No foreign key constraints or user tables.
   - Images and videos uploaded by users are stored directly as SQLite BLOBs in the database file. While functional for demo sizes, this causes database bloat if large files are uploaded.
3. **No Input Guardrails:**
   - Complaint description allows any arbitrary string without minimum word count checks or gibberish detection.

---

## 7. Phased Remediation Plan

To transform this system into an authentic, viva-defensible AI/ML college project:

- **Phase 2 (Dataset Documentation):** Document the verified dataset source, formalize the schema, clean the duplicate texts (leakage prevention), and define an authoritative mapping for fragmented departments.
- **Phase 3 (Real ML Pipeline & Model Comparison):** Build a leakage-free multi-model comparison pipeline (Baseline, Naive Bayes, Logistic Regression, Linear SVM) with stratified splitting, TF-IDF vectorization, cross-validation, and metrics export to `model_comparison.csv`.
- **Phase 4 (Live Model Integration):** Connect the winning trained model to `app.py` so typing a complaint description automatically predicts and suggests the department with real-time confidence feedback, while preserving manual override.
- **Phase 5 (Smart Prioritization):** Implement a transparent, defensible complaint priority engine taking into account severity, citizen reports count, and keyword urgency.
- **Phase 6 (Duplicate Complaint Detection):** Add similarity-based duplicate matching using TF-IDF cosine similarity against existing database complaints.
- **Phase 7 (Admin ML Intelligence Dashboard):** Upgrade the admin dashboard with actual ML evaluation metrics, confusion matrix, and model comparison visualizations, while connecting the trend charts to real database data.
- **Phase 8 (Viva Documentation):** Generate all 10 required technical documentation files, including 50+ viva defense questions and answers in Hinglish.
- **Phase 9 (End-to-End Verification):** Test and verify every module.
