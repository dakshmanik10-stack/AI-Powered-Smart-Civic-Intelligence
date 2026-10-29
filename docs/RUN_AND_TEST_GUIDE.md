# RUN & TEST GUIDE: CIVIC INTELLIGENCE SYSTEM
**Project Name:** AI-Powered Civic Complaint Intelligence System  
**Date:** October 29, 2026  

---

## 1. System Requirements & Prerequisites

- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** Python 3.10, 3.11, 3.12, or 3.13 (64-bit recommended).
- **RAM:** Minimum 2 GB (4 GB recommended).
- **Storage:** ~500 MB free space.

---

## 2. Quick Setup & Installation

### Step 1: Open Terminal in Project Directory
Navigate into the `Civic-Complaint` folder:
```powershell
cd "c:\Users\Hardware test\Desktop\civic complain\Civic-Complaint"
```

### Step 2: Install Dependencies
Run the installation command using Python's package manager:
```powershell
py -m pip install -r requirements.txt
```

*(This installs `streamlit`, `pandas`, `plotly`, `scikit-learn`, `nltk`, `joblib`)*.

---

## 3. Training & Benchmarking the Machine Learning Pipeline

To re-run the complete model training, leakage-free benchmark comparison, and generation of `.pkl` artifacts:

```powershell
py train_models.py
```

### What this does:
1. Loads `dataset/municipal_training_set_1100.csv`.
2. Standardizes 25 messy labels into 8 clear municipal departments.
3. Enforces deduplication to eliminate data leakage.
4. Performs an 80/20 stratified train/test split.
5. Fits TF-IDF and trains 4 models:
   - Majority-Class Baseline
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear Support Vector Machine (`LinearSVC`)
6. Evaluates and outputs `model_comparison.csv` and `confusion_matrices.json`.
7. Saves the winning model to `civic_model.pkl` and `tfidf_vectorizer.pkl`.

---

## 4. Running the Web Application

To launch the Streamlit frontend with AI Smart Triage, Priority Scoring, and Admin Evaluation:

```powershell
py -m streamlit run app.py
```

### Accessing the Web Portal:
- Open your browser and navigate to: **`http://localhost:8501`**

---

## 5. Default Credentials & Test Scenarios

### Admin Credentials:
- **Username:** `admin`
- **Password:** `admin123`
- **Access:** Full administrative view, GIS Dashboard, Recent Complaints status editor, and **AI Model Evaluation**.

### Citizen Demo Credentials:
- **Username:** `citizen`
- **Password:** `citizen123`
- *(Alternative: Register a new account via the **Citizen Sign up** tab; accounts persist across restarts in SQLite)*.

### Interactive AI Testing Scenarios to Try:
1. **Scenario 1 (Water Leakage):**
   - Type in Description: `"Major water pipeline burst on street corner causing severe water wastage and flooding"`
   - *Expected AI Output:*
     - 🤖 **Predicted Department:** `Water Supply & Management`
     - ⚡ **Priority:** `High / Critical`
2. **Scenario 2 (Electrical Emergency):**
   - Type in Description: `"Live electric wire sparking and hanging dangerously low over school entrance"`
   - *Expected AI Output:*
     - 🤖 **Predicted Department:** `Electrical Department`
     - ⚡ **Priority:** `Critical (Score: 9.8 / 10)` (Triggered by acute hazard keywords).
3. **Scenario 3 (Duplicate Advisory Test):**
   - Type in Description: `"A large pothole is making the road unsafe near the main crossing"`
   - *Expected AI Output:*
     - ⚠️ **Duplicate Advisory:** Flags matching complaint `#CVS-100001` with high similarity percentage!

---

## 6. Troubleshooting Common Issues

### Issue 1: `Port 8501 is already in use`
- **Fix:** Run on a specific alternative port:
  ```powershell
  py -m streamlit run app.py --server.port 8502
  ```

### Issue 2: `ModuleNotFoundError: No module named 'nltk'`
- **Fix:** Reinstall dependencies:
  ```powershell
  py -m pip install -r requirements.txt
  ```

### Issue 3: NLTK Data Not Found
- **Fix:** The app and training scripts automatically download `stopwords` and `wordnet` with `quiet=True`. If on an isolated offline network, run once:
  ```powershell
  py -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
  ```
