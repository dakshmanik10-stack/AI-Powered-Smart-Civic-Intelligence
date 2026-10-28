import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# NLTK packages download karna
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Dataset load karna (Yahan 'dataset/' folder ka naam zaroor daalein)
df = pd.read_csv('dataset/municipal_training_set_1100.csv')

# 2. Robust Cleaning: Drop missing issue descriptions and departments
df = df.dropna(subset=['issue_description', 'municipal_department'])
df = df[df['municipal_department'].astype(str).str.lower() != 'nan']

# Ensure ki wahi classes train hon jinki kam se kam 2 samples hon
class_counts = df['municipal_department'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['municipal_department'].isin(valid_classes)]

# 3. Text Preprocessing Function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower() # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Punctuation & numbers hatana
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

print("Preprocessing text data...")
df['cleaned_text'] = df['issue_description'].apply(preprocess_text)

# 4. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['municipal_department'], test_size=0.2, random_state=42
)

# 5. Feature Extraction (TF-IDF Vectorizer)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Model Training (Logistic Regression)
print("Training the Machine Learning model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Model Training Completed Successfully!")

# 7. Evaluation & Accuracy Check
y_pred = model.predict(X_test_tfidf)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Test with a Brand New Custom Complaint
sample_test = ["Street light is not working near the main square"]
cleaned_sample = preprocess_text(sample_test[0])
sample_tfidf = vectorizer.transform([cleaned_sample])
prediction = model.predict(sample_tfidf)

print("="*50)
print(f"Test Complaint: '{sample_test[0]}'")
print(f"Predicted Department: --> {prediction[0]}")
print("="*50)

# Model aur vectorizer ko save karna
joblib.dump(model, 'civic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nModel and Vectorizer saved successfully as .pkl files!")