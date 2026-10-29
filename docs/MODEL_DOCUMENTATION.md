# MACHINE LEARNING MODEL DOCUMENTATION
**Project Name:** AI-Powered Civic Complaint Intelligence System  
**Framework:** Scikit-Learn 1.9+, NLTK, NumPy, Joblib  
**Date:** October 29, 2026  

---

## 1. Overview of AI Architecture

The system utilizes an ensemble of machine learning, NLP, and information-retrieval techniques designed for real-time inference:

1. **Text Preprocessing & Feature Extraction:** NLTK WordNet Lemmatization + Sublinear TF-IDF $(N\text{-gram range } 1\text{–}2)$.
2. **Municipal Department Classifier (Supervised):** Linear Support Vector Machine (`LinearSVC`, $C=1.0$).
3. **Smart Priority Ranking Engine (Heuristic-Supervised):** Explainable multi-factor scoring model.
4. **Duplicate Complaint Matcher (Information Retrieval):** TF-IDF Cosine Similarity engine ($>45\%$ threshold).

---

## 2. Text Preprocessing & Vectorization Pipeline

### 2.1 Preprocessing Pipeline
Every complaint text string $s$ undergoes deterministic cleaning:
1. **Case Normalization:** $s \leftarrow \text{lower}(s)$
2. **Punctuation & Digit Filtering:** Non-alphabetic symbols replaced by whitespace using regular expression `[^a-zA-Z\s]`.
3. **Stopword Filtering:** NLTK standard English stopword vocabulary ($\sim 179$ words) removed to eliminate low-information tokens ("is", "near", "at", "the").
4. **Morphological Lemmatization:** NLTK `WordNetLemmatizer` maps inflectional suffixes to root dictionary lemmas (e.g. "leaking", "leaked", "leaks" $\rightarrow$ `"leak"`).
5. **Token Pruning:** Single- and two-character tokens discarded ($\text{len}(w) > 2$).

### 2.2 TF-IDF Feature Representation
Text is converted into numerical vector space using Term Frequency-Inverse Document Frequency:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- **Sublinear Term Frequency:**
  $$\text{TF}(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0$$
  This prevents words appearing 10 times from dominating 10 times more than a word appearing once.
- **Inverse Document Frequency:**
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **Feature Space:** Unigrams and Bigrams ($N$-gram range 1 to 2) with $\text{min\_df}=2$. Total vocabulary size $|V| = 799$ features.
- **Leakage Prevention:** Vectorizer is fitted strictly on $X_{train}$ and applied to $X_{test}$ and production inputs.

---

## 3. Evaluated Classification Algorithms

Four distinct classification paradigms were trained and evaluated on identical stratified splits ($N_{train}=351, N_{test}=88$):

### 3.1 Model 1: Majority-Class Baseline (`DummyClassifier`)
- **Mathematical Form:**
  $$\hat{y} = \arg\max_{c \in C} \sum_{i=1}^N \mathbb{I}(y_i = c)$$
- **Role:** Empirical lower bound benchmark. Always predicts the most frequent class ('Sanitation & Waste Management').
- **Result:** Accuracy = 20.45%, Macro-F1 = 0.0425.
- **Viva Note:** Confirms that any trained model must substantially beat 20.45% to demonstrate true learning.

### 3.2 Model 2: Multinomial Naive Bayes (`MultinomialNB`)
- **Mathematical Form:**
  Based on Bayes' theorem with conditional independence assumption:
  $$P(c | \mathbf{x}) \propto P(c) \prod_{j=1}^{|V|} P(w_j | c)^{x_j}$$
  With Laplace smoothing ($\alpha = 0.5$):
  $$P(w_j | c) = \frac{N_{c, j} + \alpha}{\sum_{k=1}^{|V|} N_{c, k} + \alpha |V|}$$
- **Hyperparameters:** $\alpha = 0.5$ (Lidstone/Laplace smoothing).
- **Result:** Accuracy = 87.50%, Macro-F1 = 0.8587, Training Latency = 9.40 ms.
- **Strengths:** Lightning-fast training and inference, minimal memory footprint.
- **Weaknesses:** Strong independence assumption fails to capture compound phrases where word order alters semantics.

### 3.3 Model 3: Multinomial Logistic Regression (`LogisticRegression`)
- **Mathematical Form:**
  Computes class posterior probabilities via the Softmax function:
  $$P(y = c | \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{k=1}^K \exp(\mathbf{w}_k^T \mathbf{x} + b_k)}$$
  Optimized via Cross-Entropy loss with $L_2$ weight regularization:
  $$J(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \log P(y_i | \mathbf{x}_i) + \frac{1}{2C} \sum_{c=1}^K \|\mathbf{w}_c\|_2^2$$
- **Hyperparameters:** $C = 1.0$, $\text{solver} = \text{'lbfgs'}$, $\text{max\_iter} = 1000$, $\text{random\_state} = 42$.
- **Result:** Accuracy = 87.50%, Macro-F1 = 0.8655, Training Latency = 70.48 ms.
- **Strengths:** Produces smooth, calibrated probabilities; interpretable coefficients.
- **Weaknesses:** Slower convergence on large feature dimensions; slightly lower margin boundary on minority classes.

### 3.4 Model 4 (Winning Model): Linear Support Vector Machine (`LinearSVC`)
- **Mathematical Form:**
  Constructs a maximum-margin separating hyperplane for each class in the 799-dimensional feature space. Solves the soft-margin primal optimization problem:
  $$\min_{\mathbf{w}_c, b_c, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}_c\|_2^2 + C \sum_{i=1}^N \xi_{i, c}$$
  Subject to:
  $$y_{i, c} (\mathbf{w}_c^T \mathbf{x}_i + b_c) \ge 1 - \xi_{i, c}, \quad \xi_{i, c} \ge 0$$
  Inference prediction uses the multi-class one-vs-rest decision function:
  $$\hat{y} = \arg\max_{c \in C} \left( \mathbf{w}_c^T \mathbf{x} + b_c \right)$$
- **Hyperparameters:** $C = 1.0$, $\text{loss} = \text{'squared\_hinge'}$, $\text{dual} = \text{'auto'}$, $\text{random\_state} = 42$.
- **Result:** Accuracy = **89.77%**, Macro Precision = **0.9125**, Macro Recall = **0.9018**, Macro-F1 = **0.8916**, Training Latency = **29.07 ms**, Prediction Latency = **2.27 ms**.
- **Why LinearSVC Won:** High-dimensional sparse TF-IDF vectors are linearly separable. SVM finds the optimal boundary that maximizes the geometric distance to the nearest training points (support vectors), yielding superior generalization on unseen phrases.

---

## 4. Smart Complaint Prioritization Engine

To avoid black-box arbitrary scores, priority is evaluated through an explainable multi-factor model:

$$\text{Priority Score} = \min\left(10.0, \max\left(1.0, S_{\text{base}} + \Delta_{\text{hazard}} + \Delta_{\text{urgency}} + \Delta_{\text{category}} + \Delta_{\text{escalation}}\right)\right)$$

Where:
- $S_{\text{base}} = 4.0$ (Default Medium priority baseline).
- $\Delta_{\text{hazard}} = +3.5$ if acute public safety hazards are detected (`spark`, `electric shock`, `explosion`, `fire`, `live wire`, `burst`, `flooding`, `open manhole`, `deep pothole`, `accident`, `poisonous`).
- $\Delta_{\text{urgency}} = +2.0$ if high-urgency keywords are detected (`leakage`, `overflow`, `stinking`, `dark street`, `blocked road`, `hazardous`).
- $\Delta_{\text{category}} = +1.5$ for high-risk infrastructure categories (`Dangerous hanging wires`, `Transformer sparks`, `Contaminated water`, `Manholes/Sewer blockages`).
- $\Delta_{\text{escalation}} = +1.5$ if $>3$ matching complaints exist nearby, or $+0.8$ if $>1$ exist.

### Priority Level Thresholds:
- **Critical (Score $\ge 8.0$):** Emergency response required immediately.
- **High (Score $6.5 \text{ to } 7.9$):** Rapid dispatch within 12–24 hours.
- **Medium (Score $4.0 \text{ to } 6.4$):** Standard municipal triage within 48–72 hours.
- **Low (Score $< 4.0$):** Routine maintenance and aesthetic civic improvements.

---

## 5. Duplicate Detection Engine (Cosine Similarity)

When a citizen drafts a complaint description $\mathbf{q}$, it is compared against all existing database complaints $D = \{\mathbf{d}_1, \dots, \mathbf{d}_M\}$ using Cosine Similarity in TF-IDF space:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}_k) = \frac{\mathbf{q} \cdot \mathbf{d}_k}{\|\mathbf{q}\|_2 \|\mathbf{d}_k\|_2} = \frac{\sum_{j=1}^{|V|} q_j d_{k, j}}{\sqrt{\sum_{j=1}^{|V|} q_j^2} \sqrt{\sum_{j=1}^{|V|} d_{k, j}^2}}$$

### Operational Rules:
- **Threshold ($\tau = 0.45$):** Pairs with similarity $\ge 45\%$ are flagged as potential duplicates.
- **Advisory Only:** Duplicates trigger an informative UI banner with matching tracking ID and status, but **never block submission or delete records**, giving citizens autonomy while preventing duplicate confusion.
