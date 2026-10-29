# EXPERIMENT RESULTS & MODEL BENCHMARKING REPORT
**Project Name:** AI-Powered Civic Complaint Intelligence System  
**Task:** Automated Municipal Department Classification from Complaint Description  
**Date:** October 29, 2026  
**Artifacts Generated:** `model_comparison.csv`, `confusion_matrices.json`, `civic_model.pkl`, `tfidf_vectorizer.pkl`  

---

## 1. Experimental Protocol & Leakage Prevention

- **Dataset Size:** 1,100 raw records (439 unique, deduplicated complaint descriptions after label mapping and cleaning).
- **Leakage Prevention Protocol:** Prior audits revealed that 653 rows in the original dataset had duplicate text descriptions. To prevent train/test contamination (data leakage), the evaluation was performed strictly on deduplicated samples so that **no complaint text appearing in the test set existed in the training set**.
- **Split Ratio:** 80% Training ($N = 351$), 20% Testing ($N = 88$).
- **Stratification:** Stratified sampling enabled across all 8 municipal departments using fixed `random_state=42`.
- **Feature Pipeline:**
  - Lowercase conversion & non-alphabetic regex removal.
  - NLTK English stopwords removal + WordNet Lemmatization.
  - TF-IDF Vectorizer fitted **exclusively on $X_{train}$** with $N$-gram range $(1, 2)$ (unigrams + bigrams), sublinear term frequency scaling, and minimum document frequency $\text{min\_df}=2$.
  - Vocabulary Size: 799 features.

---

## 2. Actual Measured Benchmark Results

All metrics below reflect real, unsimulated execution on the unseen test set ($N=88$):

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Training Time | Prediction Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 20.45% | 0.0256 | 0.1250 | 0.0425 | 0.0695 | 1.14 ms | 0.59 ms |
| **Multinomial Naive Bayes** | 87.50% | 0.8783 | 0.8565 | 0.8587 | 0.8736 | 9.40 ms | 1.73 ms |
| **Logistic Regression** ($C=1.0$) | 87.50% | 0.8952 | 0.8661 | 0.8655 | 0.8789 | 70.48 ms | 1.05 ms |
| **Linear Support Vector Machine (LinearSVC)** | **89.77%** | **0.9125** | **0.9018** | **0.8916** | **0.9022** | **29.07 ms** | **2.27 ms** |

---

## 3. Class-Wise Performance (Winning Model: LinearSVC)

Overall Test Accuracy: **89.77%** | Macro-F1: **0.8916** | Weighted-F1: **0.9022**

| Standardized Department | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **Drainage Authority** | 1.00 | 0.86 | 0.92 | 7 |
| **Electrical Department** | 1.00 | 1.00 | 1.00 | 12 |
| **Parks & Community Services** | 0.50 | 1.00 | 0.67 | 7 |
| **Public Safety & Enforcement** | 0.90 | 0.90 | 0.90 | 10 |
| **Roads & Infrastructure** | 0.90 | 0.60 | 0.72 | 15 |
| **Sanitation & Waste Management** | 1.00 | 1.00 | 1.00 | 18 |
| **Traffic Department** | 1.00 | 0.86 | 0.92 | 7 |
| **Water Supply & Management** | 1.00 | 1.00 | 1.00 | 12 |

---

## 4. Key Engineering Insights & Viva Defense Analysis

1. **Why LinearSVC Won:**
   - Linear Support Vector Machines construct a maximum-margin separating hyperplane in high-dimensional sparse TF-IDF space (799 dimensions). Because text data with n-grams is highly linearly separable, SVM outperforms probabilistic models by penalizing margin violations without being influenced by feature correlation.
2. **Why Majority Baseline Scored 20.45%:**
   - The majority class ('Sanitation & Waste Management') represents roughly 20.9% of the dataset. A naive classifier predicting only Sanitation achieves ~20.4% accuracy, proving that our 89.77% model delivers an authentic **~69.3% accuracy gain** above chance.
3. **Failure Case Analysis (Roads vs Parks):**
   - 'Parks & Community Services' achieved lower precision (0.50) because complaints describing "cracked pedestrian pavement in community parks" overlap with 'Roads & Infrastructure'.
   - This provides a genuine, realistic talking point for college viva: in municipal governance, dual-jurisdiction issues (e.g. road repair inside a municipal park) require human administrative review.
