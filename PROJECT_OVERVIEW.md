# PROJECT OVERVIEW: AI-POWERED SMART CIVIC INTELLIGENCE SYSTEM
**Course/Context:** Final Year / Academic College Project  
**Author:** Student Engineering Team  
**Tech Stack:** Python 3.13, Streamlit, Scikit-Learn, NLTK, SQLite3, Plotly, Pandas, Joblib  

---

## 1. Problem Statement

In contemporary municipal governance, citizen grievance redressal is plagued by three fundamental bottlenecks:
1. **Manual Triage & Routing Delays:** When citizens submit complaints through standard web forms, administrative personnel must manually read, interpret, and forward each ticket to the relevant municipal department (e.g. Roads, Sanitation, Water Supply, Electricity, Traffic). This manual step causes significant administrative backlogs, delays, and frequent misrouting.
2. **Lack of Objective Priority Ranking:** Municipal authorities treat complaints on a crude first-come-first-served basis rather than by acute public safety risk. A life-threatening open live electrical wire or severe road crater is frequently queued behind benign aesthetic issues like park lawn trimming.
3. **Complaint Redundancy & Duplicate Submissions:** During civic infrastructure breakdowns (such as water pipeline bursts or localized transformer sparks), dozens of neighboring citizens file separate complaints for the identical issue. Municipal officers waste time processing duplicated tickets rather than resolving the physical incident.

---

## 2. Project Objectives

1. **Automated NLP Department Classification:** Train and deploy a high-accuracy, leakage-free text classifier that reads citizen complaint descriptions in natural language and automatically routes them to the correct municipal department with over 89% accuracy.
2. **Explainable Smart Prioritization:** Build an objective, transparent priority engine that scores complaints from 1.0 to 10.0 and assigns priority tiers (`Critical`, `High`, `Medium`, `Low`) based on public hazard keywords, category impact weights, and citizen escalation counts.
3. **Similarity-Based Duplicate Detection:** Implement real-time TF-IDF Cosine Similarity matching to detect existing open grievances at submission time, alerting the citizen while keeping records clean.
4. **Interactive Citizen & Administrative GIS Portal:** Provide an intuitive dual-portal interface where citizens can report grievances, upload media, and track live status, while administrators manage complaints, view spatial GIS charts, and examine empirical AI model evaluation metrics.

---

## 3. High-Level Architecture & Workflow

```text
[ Citizen Input ]
       │
       ▼ (Natural Language Description)
[ NLP Preprocessing Pipeline ] (Lowercase, Regex, Stopwords, WordNet Lemmatization)
       │
       ▼ (TF-IDF Vectorization: 799 n-grams)
[ Multi-Model AI Engine ]
  ├── 1. Department Routing (LinearSVC Classifier: 89.8% Accuracy)
  ├── 2. Smart Prioritization (Keyword Risk + Multi-Report Multiplier)
  └── 3. Duplicate Detection (Cosine Similarity against Database Corpus)
       │
       ▼ (AI Recommendations & Previews)
[ Citizen Confirmation & Storage ] ──► [ SQLite Database (`civic_complaints.db`) ]
                                                   │
                                                   ▼
                                      [ Admin Review & Analytics ]
                                        ├── Recent Complaints & Status Updates
                                        ├── Operational GIS Dashboard
                                        └── ML Evaluation & Confusion Matrix
```

---

## 4. Current Implementation Status

- **Citizen Portal:** Fully functional with interactive AI Smart Triage card, auto-suggested categories, duplicate warnings, and media upload.
- **Citizen Dashboard:** Tracks complaints filed by citizen ID, displaying dynamic Status badges and colored Priority badges (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
- **Admin Portal:** Live editable table supporting inline status transitions (`Pending` ➔ `In Progress` ➔ `Resolved` ➔ `Rejected`), persisting to SQLite.
- **Admin GIS & Overview Dashboard:** Service-level indicator gauge, authentic complaints timeline from SQLite records, category breakdown, and priority distribution pie chart.
- **Dedicated ML Evaluation Dashboard:** Benchmark table across 4 models (`Majority Baseline`, `MultinomialNB`, `LogisticRegression`, `LinearSVC`), latency analysis, and 8-department interactive Confusion Matrix.
- **Persistent Authentication:** Dynamic citizen registrations persist in SQLite `users` table across reboots.

---

## 5. Known Limitations & Future Roadmap

1. **English Language Specificity:** The current NLP pipeline is trained on English text. Future releases will integrate multilingual embeddings (e.g. IndicBERT / XLM-RoBERTa) to support Hindi, Hinglish, and regional languages.
2. **Automated Vision Inspection:** Currently photos/videos are attached for human verification. In future work, a fine-tuned YOLOv8 model can automatically detect potholes and garbage overflow directly from uploaded images.
3. **Official Municipality Integration:** The current application operates as an independent demonstration system with local SQLite storage rather than an official government API gateway.
