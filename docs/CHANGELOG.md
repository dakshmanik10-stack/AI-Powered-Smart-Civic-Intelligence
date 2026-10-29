# PROJECT CHANGELOG & MIGRATION HISTORY
**Project Name:** AI-Powered Civic Complaint Intelligence System  
**Date:** October 29, 2026  

---

## [Version 2.0.0] - AI System Upgrade & Academic Standardization (October 2026)

### 1. Machine Learning & Intelligence Engine
- **Added `ml_engine.py`:** Singleton ML inference engine providing real-time department classification, priority scoring, and duplicate matching.
- **Added `train_models.py`:** Empirical benchmarking script that trains and evaluates 4 algorithms (Majority Baseline, MultinomialNB, Logistic Regression, LinearSVC) on leakage-free deduplicated data.
- **Model Comparison Export:** Generated `model_comparison.csv` and `confusion_matrices.json`.
- **Winning Production Model:** Linear Support Vector Machine (`LinearSVC`) selected with **89.77% accuracy**, **0.8916 Macro-F1**, and **2.27 ms prediction latency**.
- **Model Artifacts:** Saved optimized `civic_model.pkl` and `tfidf_vectorizer.pkl` (799 unigram/bigram features).

### 2. Citizen Portal Enhancements
- **Reactive AI Smart Triage:** As citizens type complaint descriptions ($\ge 10$ chars), the UI dynamically predicts the municipal department with confidence score and priority assessment.
- **Automated Category Mapping:** AI automatically maps predicted departments to relevant category pills while preserving full citizen manual override.
- **Duplicate Complaint Advisory:** Computes TF-IDF Cosine Similarity against all stored complaints; alerts citizens to matching existing reports without blocking submissions.
- **Enhanced Complaint Cards:** Added dynamic priority badges (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) and assigned authority tags.

### 3. Administrator Portal & GIS Dashboard
- **Dedicated ML Evaluation Dashboard:** Added third administrative navigation tab displaying model comparison bar charts, training latency metrics, and interactive 8-department Confusion Matrix.
- **Priority Escalation Badges:** Added Priority level and numeric score columns to the admin recent complaints table.
- **Authentic Database Timeline:** Replaced hardcoded February 2023 mock trend array with actual complaint filing counts grouped by date from SQLite.
- **Priority Distribution:** Added interactive donut pie chart showing real-time priority breakdown.

### 4. Database & Security Improvements
- **Schema Migration:** Added `predicted_department`, `priority_level`, `priority_score`, `priority_reason`, and `matched_complaint_id` columns to `complaints` table in `civic_complaints.db`.
- **Persistent Citizen Authentication:** Added `users` table to SQLite so citizen accounts persist permanently across browser reloads and server restarts.
- **Historical Data Backfill:** Automatically backfilled existing database records with AI predictions and priority scores.

### 5. Documentation & Academic Governance
- **Created Comprehensive Documentation Suite in `docs/` and Root:**
  1. `PROJECT_OVERVIEW.md`
  2. `PROJECT_AUDIT.md`
  3. `DATASET_DOCUMENTATION.md`
  4. `MODEL_DOCUMENTATION.md`
  5. `EXPERIMENT_RESULTS.md`
  6. `TECHNOLOGY_JUSTIFICATION.md`
  7. `ARCHITECTURE.md`
  8. `VIVA_QUESTIONS.md` (55+ questions with beginner-friendly Hinglish answers)
  9. `RUN_AND_TEST_GUIDE.md`
  10. `CHANGELOG.md`
- **Created `requirements.txt`:** Explicitly documented package versions for environment reproducibility.

---

## [Version 1.0.0] - Initial Prototype (Friend's Legacy State)
- Basic Streamlit layout with login, signup, complaint filing, and admin review.
- Standalone `main.py` training a single Logistic Regression model on raw CSV with 653 duplicate texts.
- Untrained `app.py`: No connection between frontend and trained `.pkl` models.
- Ephemeral in-memory citizen user accounts (`st.session_state`).
- Hardcoded mock data in admin trend line chart.
- No priority scoring or duplicate detection.
