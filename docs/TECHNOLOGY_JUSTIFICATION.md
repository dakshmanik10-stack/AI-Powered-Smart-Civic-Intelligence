# TECHNOLOGY JUSTIFICATION & TRADE-OFF ANALYSIS
**Project Name:** AI-Powered Civic Complaint Intelligence System  
**Date:** October 29, 2026  

---

## 1. Executive Summary

Every software component and machine learning library in this project was selected based on computational efficiency, deployment simplicity, interpretability, and academic reproducibility. The table below details why each technology was chosen, what alternatives were considered, and the trade-offs involved.

---

## 2. Component-by-Component Justification

| Technology / Library | Role in Project | Why Chosen? (Advantages) | Rejected Alternatives | Trade-Offs & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Python 3.13** | Core Programming Language | Universal ecosystem for AI/ML, rich NLP libraries, rapid prototyping, and cross-platform compatibility. | C++, Java, Node.js | Slightly slower execution speed than compiled C++, but Python’s C-optimized libraries (NumPy, Scikit-Learn) eliminate CPU bottlenecks for text inference. |
| **Streamlit** | Web Application Framework | High-velocity Python-native UI; binds reactive states without needing separate REST APIs or JavaScript frontend code. | Flask, Django, FastAPI + React | React+FastAPI offers higher multi-tenant customizability, but requires maintaining separate codebases and deployment stacks. Streamlit provides immediate full-stack interactivity for college evaluation. |
| **Scikit-Learn** | Machine Learning Framework | Production-tested algorithms (SVM, Logistic Regression, Naive Bayes, TF-IDF), minimal dependency overhead, fast training, CPU execution. | PyTorch, TensorFlow, HuggingFace Transformers | Deep Transformer models (BERT) require 1–2 GB GPU memory and take seconds per inference on CPU. LinearSVC delivers 89.8% accuracy in just 2.2 ms without specialized hardware. |
| **NLTK (Natural Language Toolkit)** | NLP Tokenization & Lemmatization | Standard, lightweight WordNet lemmatizer and curated stopword lists; deterministic and easily explained in viva. | spaCy, Stanza | spaCy is powerful but has a heavy binary memory footprint (100+ MB model download). NLTK accomplishes dictionary-based lemmatization with zero runtime bloat. |
| **SQLite3** | Relational Database Engine | Zero-configuration serverless database bundled directly in Python; single file (`civic_complaints.db`); ACID compliant. | PostgreSQL, MySQL, MongoDB | Postgres requires installing and running a continuous background daemon service. SQLite satisfies all academic demo persistence requirements with zero external setup. |
| **Plotly (Express & Graph Objects)** | Data Visualization | Generates interactive HTML5 charts (zoom, hover tooltips, responsive layout); native Streamlit rendering. | Matplotlib, Seaborn | Matplotlib renders static raster PNG images. Plotly provides dynamic tooltips and responsive gauges essential for a modern administrative dashboard. |
| **Pandas** | Tabular Data Processing | Industry-standard DataFrame operations, seamless CSV manipulation, grouped aggregations, and integration with Streamlit `data_editor`. | Polars, Raw Python Dicts | Polars is faster on multi-gigabyte datasets, but on a 1,100-row municipal dataset, Pandas is universally compatible and standard. |
| **Joblib** | Model Serialization | Optimized binary serialization for NumPy-heavy machine learning models; faster loading than vanilla `pickle`. | Pickle, ONNX | Standard `pickle` struggles with large sparse matrices. Joblib handles disk caching and NumPy arrays with maximum I/O efficiency. |

---

## 3. Architecture Comparison: Lightweight ML vs Deep Learning

In modern AI engineering, matching model complexity to problem constraints is a primary design virtue:

```text
Feature                  Our Stack (LinearSVC + TF-IDF)     Heavy Deep Learning (BERT / LLM)
─────────────────────────────────────────────────────────────────────────────────────────────
Inference Latency        ~2.27 milliseconds                 250–800 milliseconds (CPU)
Memory Requirement       < 10 MB RAM                        ~1,500 MB RAM + GPU VRAM
Hardware Dependency      Runs on any standard PC            Requires CUDA / High-end GPU
Cold Start Time          Instantaneous (< 0.1s)             5–10 seconds model load
Accuracy on Dataset      89.77%                             ~91–93% (Marginal gain)
Interpretability         Full support vectors & weights     Black box neural representations
Maintenance Overhead     Single .pkl file (127 KB)          Multi-gigabyte checkpoint files
```

**Viva Takeaway:** The selection of `LinearSVC` with `TF-IDF` represents an optimal, defensible engineering decision: it achieves nearly the same real-world classification accuracy as large neural networks while operating with **100x lower latency and zero GPU requirements**.
