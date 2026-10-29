import re
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK packages
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "civic_model.pkl"
VEC_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
CONFUSION_PATH = BASE_DIR / "confusion_matrices.json"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Preprocesses raw complaint text using lowercase, regex tokenization, stopword removal, and lemmatization."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(cleaned)

class MLEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if MODEL_PATH.exists() and VEC_PATH.exists():
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VEC_PATH)
            else:
                self.model = None
                self.vectorizer = None
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            self.model = None
            self.vectorizer = None

    def predict_department(self, description: str):
        """
        Predicts the responsible municipal department for a complaint description.
        Returns: (predicted_department, confidence_indicator, status_message)
        """
        if not description or len(description.strip()) < 10:
            return None, 0.0, "Description too short for AI analysis (min 10 chars required)."

        if self.model is None or self.vectorizer is None:
            self._load_artifacts()
            if self.model is None or self.vectorizer is None:
                return None, 0.0, "AI Model artifacts not found. Please run train_models.py."

        cleaned = preprocess_text(description)
        if not cleaned:
            return None, 0.0, "Description contains no meaningful keywords."

        X_vec = self.vectorizer.transform([cleaned])

        # LinearSVC decision function
        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X_vec)
            if decision.ndim == 1:
                pred_idx = np.argmax([decision[0], -decision[0]])
                pred_class = self.model.classes_[pred_idx]
                score = float(np.abs(decision[0]))
            else:
                pred_idx = np.argmax(decision[0])
                pred_class = self.model.classes_[pred_idx]
                # Softmax-approximated confidence score for UI visualization
                exp_scores = np.exp(decision[0] - np.max(decision[0]))
                probs = exp_scores / np.sum(exp_scores)
                score = float(probs[pred_idx])
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_vec)[0]
            pred_idx = np.argmax(probs)
            pred_class = self.model.classes_[pred_idx]
            score = float(probs[pred_idx])
        else:
            pred_class = self.model.predict(X_vec)[0]
            score = 0.85

        return pred_class, round(score, 2), "Prediction generated successfully."

    def calculate_priority(self, description: str, category: str, report_count: int = 1):
        """
        Calculates an explainable priority score and level based on public safety keywords,
        infrastructure severity, category risk weight, and duplicate report count.
        Returns: (priority_level, priority_score, reason)
        """
        text_lower = description.lower()
        score = 4.0 # Base score (Medium priority)
        reasons = []

        # High-severity / public safety triggers
        critical_keywords = [
            "spark", "electric shock", "transformer explosion", "fire", "live wire",
            "burst", "flooding", "open manhole", "deep pothole", "accident",
            "casualty", "injury", "poisonous", "contaminated water", "school entrance"
        ]
        high_keywords = [
            "leakage", "overflow", "stinking", "dark street", "traffic jam",
            "blocked road", "danger", "hazardous", "urgent", "broken pipe"
        ]

        crit_matches = [k for k in critical_keywords if k in text_lower]
        if crit_matches:
            score += 3.5
            reasons.append(f"Critical safety hazard keywords detected: {', '.join(crit_matches)}")

        high_matches = [k for k in high_keywords if k in text_lower and k not in crit_matches]
        if high_matches:
            score += 2.0
            reasons.append(f"High urgency keywords detected: {', '.join(high_matches)}")

        # Category intrinsic weights
        high_risk_categories = [
            "Dangerous hanging wires", "Transformer sparks", "Contaminated water",
            "Manholes/Sewer blockages", "Faulty traffic lights"
        ]
        if category in high_risk_categories:
            score += 1.5
            reasons.append(f"High-impact civic category: {category}")

        # Repeated citizen complaints escalation
        if report_count > 3:
            score += 1.5
            reasons.append(f"High citizen impact: {report_count} nearby citizen reports filed")
        elif report_count > 1:
            score += 0.8
            reasons.append(f"Multi-citizen report ({report_count} residents affected)")

        # Cap score between 1.0 and 10.0
        final_score = min(max(round(score, 1), 1.0), 10.0)

        # Map to priority levels
        if final_score >= 8.0:
            level = "Critical"
        elif final_score >= 6.5:
            level = "High"
        elif final_score >= 4.0:
            level = "Medium"
        else:
            level = "Low"

        reason_summary = " | ".join(reasons) if reasons else "Standard civic grievance with no acute hazards detected."
        return level, final_score, reason_summary

    def check_duplicate_complaints(self, new_description: str, existing_reports: list, threshold: float = 0.50):
        """
        Uses TF-IDF Cosine Similarity to detect similar existing complaints in the database.
        Returns: list of dicts with {id, description, location, similarity_score}
        """
        if not new_description or len(new_description.strip()) < 10 or not existing_reports:
            return []

        cleaned_new = preprocess_text(new_description)
        if not cleaned_new:
            return []

        existing_descriptions = [preprocess_text(r.get("description", "")) for r in existing_reports]
        
        # Fit a temporary TF-IDF on this comparison corpus
        corpus = [cleaned_new] + existing_descriptions
        try:
            temp_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            tfidf_mat = temp_vec.fit_transform(corpus)
            sim_scores = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:])[0]

            matches = []
            for idx, score in enumerate(sim_scores):
                if score >= threshold:
                    matched_report = existing_reports[idx]
                    matches.append({
                        "id": matched_report.get("id"),
                        "category": matched_report.get("category"),
                        "location": matched_report.get("location"),
                        "description": matched_report.get("description"),
                        "status": matched_report.get("status"),
                        "similarity_score": round(float(score) * 100, 1)
                    })

            # Sort by similarity descending
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            return matches[:3] # return top 3 matches
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return []

    def get_benchmark_results(self):
        """Loads model_comparison.csv and confusion_matrices.json if present."""
        df_comparison = None
        confusion_data = None
        if COMPARISON_PATH.exists():
            df_comparison = pd.read_csv(COMPARISON_PATH)
        if CONFUSION_PATH.exists():
            with open(CONFUSION_PATH, "r") as f:
                confusion_data = json.load(f)
        return df_comparison, confusion_data

# Singleton instance
ml_engine = MLEngine()
