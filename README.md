# 🏙️ AI-Powered Smart Civic Intelligence System

An end-to-end Machine Learning and Natural Language Processing (NLP) system for automated municipal citizen grievance triage, predictive department routing, explainable priority scoring, and duplicate complaint detection.

Built with **Python 3.13**, **Streamlit**, **Scikit-Learn**, **NLTK**, **SQLite3**, and **Plotly**.

---

## 📌 Project Overview

In traditional municipal grievance redressal systems, complaints filed by citizens are manually read, categorized, and assigned to municipal departments. This creates three critical bottlenecks:
1. **Administrative Delays:** Manual triage takes days or weeks and frequently results in misrouted tickets.
2. **Subjective Priority:** Urgent hazards (e.g., exposed live electrical wires, sewer cave-ins) sit in the same queue as routine maintenance requests.
3. **Duplicate Ticket Overload:** When localized infrastructure fails (e.g., a burst water main), dozens of nearby residents file identical tickets, overwhelming city staff.

This system resolves these issues through an intelligent, leakage-free NLP pipeline that predicts responsible departments in real time, computes explainable multi-factor hazard scores, and flags existing duplicate grievances.

---

## 🚀 Key Features

- **🤖 Automated Department Classification:** Evaluates natural language descriptions and routes them to 8 standardized municipal departments with **89.77% test accuracy** and **0.8916 Macro-F1**.
- **⚡ Explainable Priority Engine:** Scores complaints from 1.0 to 10.0 and assigns priority tiers (`Critical`, `High`, `Medium`, `Low`) based on acute public safety triggers, category impact weights, and citizen escalation counts.
- **🔍 TF-IDF Duplicate Detection:** Real-time cosine similarity search across existing open tickets ($>45\%$ match threshold) to advise citizens of open neighboring complaints before filing.
- **👥 Role-Based Portals:**
  - **Citizen Portal:** User registration, complaint submission with image/video upload, live status tracking, and AI Smart Triage previews.
  - **Admin Portal:** Inline status updater (`Pending` ➔ `In Progress` ➔ `Resolved` ➔ `Rejected`), real-time operational GIS timeline, and service-level gauge.
- **📊 Dedicated ML Evaluation Dashboard:** Live administrative tab presenting multi-model empirical benchmarks, latency comparisons, and an interactive $8 \times 8$ Confusion Matrix.
- **🔒 Persistent Local Storage:** Zero-configuration local SQLite database (`civic_complaints.db`) maintaining complaints, GIS coordinates, and user accounts.

---

## 🏗️ System Architecture

```text
[ Citizen Complaint Input ]
            │
            ▼ (Raw Natural Language Description)
[ NLP Preprocessing Pipeline ] (Lowercase, Regex Tokenization, Stopwords, WordNet Lemmatization)
            │
            ▼ (TF-IDF Vectorizer: 799 unigram/bigram n-grams)
[ Multi-Model Intelligence Engine ]
  ├── 1. Department Routing (Linear Support Vector Machine: 89.8% Accuracy)
  ├── 2. Smart Priority Engine (Hazard Detection + Multi-Citizen Multiplier)
  └── 3. Duplicate Detection (TF-IDF Cosine Similarity against Database Corpus)
            │
            ▼ (Real-time Triage Insights)
[ Citizen Confirmation & Storage ] ──► [ Local SQLite Database (`civic_complaints.db`) ]
                                                    │
                                                    ▼
                                       [ Admin GIS & ML Dashboard ]
                                         ├── Recent Complaints Status Editor
                                         ├── Real Database Timeline Trend
                                         └── Model Benchmarks & Confusion Matrix
```

---

## 📊 Empirical Model Benchmarks

All models were evaluated on identical, stratified, leakage-free unseen test splits ($N_{train}=351, N_{test}=88$):

| Algorithm | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | 20.45% | 0.0256 | 0.1250 | 0.0425 | 0.0695 | 1.14 ms |
| **Multinomial Naive Bayes** | 87.50% | 0.8783 | 0.8565 | 0.8587 | 0.8736 | 9.40 ms |
| **Logistic Regression** ($C=1.0$) | 87.50% | 0.8952 | 0.8661 | 0.8655 | 0.8789 | 70.48 ms |
| **Linear Support Vector Machine (LinearSVC)** | **89.77%** | **0.9125** | **0.9018** | **0.8916** | **0.9022** | **29.07 ms** |

*Inference latency: ~2.27 ms per query on standard CPU.*

---

## 📁 Repository Structure

```text
├── app.py                         # Streamlit Web Application (Citizen & Admin Portals)
├── ml_engine.py                   # Singleton ML Inference, Priority, & Similarity Engine
├── train_models.py                # Multi-model benchmarking and training pipeline
├── test_system.py                 # Automated 8-point verification test suite
├── cluster_complaints.py          # Unsupervised semantic clustering experiment
├── inspect_data.py                # Dataset inspection utility
├── EDA_and_Preprocessing.ipynb    # Exploratory Data Analysis notebook
├── civic_model.pkl                # Production LinearSVC classifier artifact
├── tfidf_vectorizer.pkl           # Production TF-IDF vectorizer artifact
├── model_comparison.csv           # Measured benchmarking metrics
├── confusion_matrices.json        # Class-wise confusion matrix data
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules for DB, cache, and secrets
├── dataset/
│   └── municipal_training_set_1100.csv  # 1,100 municipal complaint benchmark dataset
└── docs/                          # Comprehensive Academic & Viva Documentation
    ├── PROJECT_OVERVIEW.md        # Problem statement, objectives, architecture
    ├── PROJECT_AUDIT.md           # Baseline audit and data leakage discovery
    ├── DATASET_DOCUMENTATION.md   # Dataset schema, source origin, label mapping
    ├── MODEL_DOCUMENTATION.md     # Algorithms, mathematical intuition, hyperparameters
    ├── EXPERIMENT_RESULTS.md      # Detailed metrics, split protocol, class-wise reports
    ├── TECHNOLOGY_JUSTIFICATION.md# Trade-off analysis (ML vs DL, Streamlit vs React)
    ├── ARCHITECTURE.md            # Detailed data flows and Mermaid sequence diagrams
    ├── VIVA_QUESTIONS.md          # 55+ viva questions with beginner-friendly Hinglish answers
    ├── RUN_AND_TEST_GUIDE.md      # Detailed setup, commands, and test scenarios
    └── CHANGELOG.md               # Version 1.0 to Version 2.0 migration history
```

---

## 🛠️ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dakshmanik10-stack/AI-Powered-Smart-Civic-Intelligence.git
cd AI-Powered-Smart-Civic-Intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
Verify that the model, vectorizer, priority engine, duplicate matcher, and database operate correctly:
```bash
python test_system.py
```
*(All 8 test suites should report `[PASS]`)*.

### 4. Re-Train & Benchmark Models (Optional)
To re-evaluate the candidate classifiers and update `model_comparison.csv`:
```bash
python train_models.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔑 Demo Login Credentials

- **Admin Portal:**
  - **Username:** `admin`
  - **Password:** `admin123`
- **Citizen Demo:**
  - **Username:** `citizen`
  - **Password:** `citizen123`
  - *(New citizen accounts can also be created via the **Citizen Sign up** tab; accounts persist across restarts in SQLite)*.

---

## 📑 Academic & Viva Documentation

For project defense and viva preparation, refer to the detailed reports in the [`docs/`](./docs) directory:
- [**55+ Viva Questions & Answers**](./docs/VIVA_QUESTIONS.md)
- [**Machine Learning Model Documentation**](./docs/MODEL_DOCUMENTATION.md)
- [**Dataset Source & Leakage Audit**](./docs/DATASET_DOCUMENTATION.md)
- [**Technology Trade-off Justification**](./docs/TECHNOLOGY_JUSTIFICATION.md)
- [**System Architecture & Data Flows**](./docs/ARCHITECTURE.md)

---

## ⚖️ Notice & Academic Disclaimer

This project is developed for educational and academic research demonstration. It operates independently with local SQLite persistence and is not directly affiliated with or connected to any municipal government grievance portal.