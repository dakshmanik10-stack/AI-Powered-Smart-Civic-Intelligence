import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# 1. Dataset load karein
df = pd.read_csv('dataset/municipal_training_set_1100.csv')

# Missing values remove karein
df = df.dropna(subset=['issue_description'])

# Sirf pehli 200 complaints le rahe hain testing aur speed ke liye
texts = df['issue_description'].head(200).tolist()

print("Loading Sentence Transformer model...")
# Model load kar rahe hain jo text ko numbers (embeddings) mein badlega
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating sentence embeddings...")
embeddings = model.encode(texts)

print("Clustering complaints into common problem groups...")
# Maan lijiye hum inhein 10 alag-alag problem groups mein baant rahe hain
num_clusters = 10
clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
clustering_model.fit(embeddings)
cluster_labels = clustering_model.labels_

# DataFrame mein cluster labels add karein
df_subset = df.head(200).copy()
df_subset['cluster_id'] = cluster_labels

# Har cluster ke kuch examples dekhte hain
for i in range(num_clusters):
    print(f"\n--- PROBLEM GROUP / CLUSTER {i+1} ---")
    group_complaints = df_subset[df_subset['cluster_id'] == i]['issue_description'].tolist()
    for c in group_complaints[:3]:  # Har group ki pehli 3 complaints
        print(f"  - {c}")

from sklearn.feature_extraction.text import TfidfVectorizer

print("\n" + "="*50)
print("📝 GENERATING PROBLEM SUMMARIES")
print("="*50)

# TF-IDF Vectorizer initialize karein (Sirf English stop words hatayenge)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5)

for i in range(num_clusters):
    # Us specific cluster ki saari complaints nikalein
    group_complaints = df_subset[df_subset['cluster_id'] == i]['issue_description'].tolist()
    
    if len(group_complaints) > 0:
        # TF-IDF fit karein us cluster ke text par
        tfidf_matrix = vectorizer.fit_transform(group_complaints)
        
        # Sabse important keywords nikalein
        keywords = vectorizer.get_feature_names_out()
        
        print(f"\n🔹 Cluster {i+1} Summary:")
        print(f"   Total Complaints in this group: {len(group_complaints)}")
        print(f"   Key Problem Words: {', '.join(keywords)}")
        print(f"   Example Complaint: '{group_complaints[0][:80]}...'")