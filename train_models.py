import time
import json
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Ensure NLTK packages
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 1. Authoritative Department Mapping
DEPARTMENT_MAPPING = {
    'Sanitation': 'Sanitation & Waste Management',
    'Waste Management': 'Sanitation & Waste Management',
    'Water Department': 'Water Supply & Management',
    'Water Management': 'Water Supply & Management',
    'Water Services': 'Water Supply & Management',
    'Waterworks': 'Water Supply & Management',
    'Roads & Highways': 'Roads & Infrastructure',
    'Road Maintenance': 'Roads & Infrastructure',
    'Roads': 'Roads & Infrastructure',
    'Public Works': 'Roads & Infrastructure',
    'Traffic Department': 'Traffic Department',
    'Drainage Authority': 'Drainage Authority',
    'Electrical Department': 'Electrical Department',
    'Electrical Services': 'Electrical Department',
    'Electrical': 'Electrical Department',
    'Public Safety': 'Public Safety & Enforcement',
    'Noise Control': 'Public Safety & Enforcement',
    'Code Enforcement': 'Public Safety & Enforcement',
    'Parks & Recreation': 'Parks & Community Services',
    'Parks Department': 'Parks & Community Services',
    'Community Services': 'Parks & Community Services',
    'Community Affairs': 'Parks & Community Services',
    'Event Management': 'Parks & Community Services',
}

# 2. Text Preprocessor
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(cleaned)

def main():
    print("=" * 60)
    print("PHASE 3: DEPARTMENT CLASSIFICATION EXPERIMENTS & BENCHMARKING")
    print("=" * 60)

    # Load dataset
    csv_path = 'dataset/municipal_training_set_1100.csv'
    raw_df = pd.read_csv(csv_path)
    print(f"Loaded raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # Map department
    raw_df['standard_department'] = raw_df['municipal_department'].map(DEPARTMENT_MAPPING)
    
    # Drop rows without valid mapped department or description
    valid_df = raw_df.dropna(subset=['issue_description', 'standard_department']).copy()
    print(f"Rows after dropping unmapped/null departments: {len(valid_df)}")

    # Leakage Prevention: Deduplicate by unique issue_description
    clean_df = valid_df.drop_duplicates(subset=['issue_description']).copy()
    print(f"Unique, clean complaint descriptions (Leakage-Free): {len(clean_df)}")
    print("\nClass distribution in clean unique dataset:")
    print(clean_df['standard_department'].value_counts())

    # Text Preprocessing
    clean_df['cleaned_text'] = clean_df['issue_description'].apply(clean_text)

    # Stratified Train-Test Split (80% Train, 20% Test, random_state=42)
    X = clean_df['cleaned_text']
    y = clean_df['standard_department']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Fit TF-IDF Vectorizer ONLY on training set (Preventing feature/vocabulary leakage)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF Vocabulary Size (Unigrams + Bigrams): {len(vectorizer.get_feature_names_out())}")

    # Models to compare
    models = {
        "Majority-Class Baseline": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear Support Vector Machine": LinearSVC(C=1.0, random_state=42, dual='auto')
    }

    results = []
    confusion_matrices = {}

    for name, clf in models.items():
        print(f"\nEvaluating: {name} ...")
        
        # Training time
        t0 = time.perf_counter()
        clf.fit(X_train_tfidf, y_train)
        train_time = (time.perf_counter() - t0) * 1000 # ms

        # Prediction time
        t1 = time.perf_counter()
        y_pred = clf.predict(X_test_tfidf)
        pred_time = (time.perf_counter() - t1) * 1000 # ms

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        macro_p = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_r = recall_score(y_test, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

        confusion_matrices[name] = {
            "classes": sorted(y.unique()),
            "matrix": cm.tolist()
        }

        print(f"  Accuracy:    {acc*100:.2f}%")
        print(f"  Macro-F1:    {macro_f1:.4f}")
        print(f"  Weighted-F1: {weighted_f1:.4f}")
        print(f"  Train Time:  {train_time:.2f} ms | Pred Time: {pred_time:.2f} ms")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro Precision": round(macro_p, 4),
            "Macro Recall": round(macro_r, 4),
            "Macro F1": round(macro_f1, 4),
            "Weighted F1": round(weighted_f1, 4),
            "Train Time (ms)": round(train_time, 2),
            "Pred Time (ms)": round(pred_time, 2),
            "Train Samples": len(X_train),
            "Test Samples": len(X_test)
        })

    # Save comparison to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY (Saved to model_comparison.csv):")
    print("="*60)
    print(results_df.to_string(index=False))

    # Save confusion matrices to JSON
    with open("confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # Determine winning model based on Macro-F1 and Accuracy
    best_idx = results_df['Macro F1'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_clf = models[best_model_name]
    print(f"\nWinning Model Selected: {best_model_name} (Macro F1 = {results_df.loc[best_idx, 'Macro F1']})")

    # Retrain/Save winning model and vectorizer
    joblib.dump(best_clf, 'civic_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Saved production artifacts: civic_model.pkl and tfidf_vectorizer.pkl")

    # Detailed report of best model
    print(f"\n--- Detailed Classification Report for {best_model_name} ---")
    y_pred_best = best_clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred_best, zero_division=0))

    # Quick test on unseen sample
    test_samples = [
        "Major water pipeline has burst near the market creating flooding.",
        "Deep pothole on main road causing severe accident risk for two wheelers.",
        "Garbage pile rotting on the street corner attracting stray animals.",
        "Street light pole spark and wire hanging dangerously low over footpath."
    ]
    print("--- Real Unseen Test Prediction Check ---")
    for s in test_samples:
        s_clean = clean_text(s)
        s_vec = vectorizer.transform([s_clean])
        pred = best_clf.predict(s_vec)[0]
        print(f"Complaint: '{s}'")
        print(f"--> Predicted Department: {pred}\n")

if __name__ == '__main__':
    main()
