import pandas as pd

# 1. Dataset load karein
df = pd.read_csv('dataset/municipal_training_set_1100.csv')

print("="*50)
print("📊 DATASET SUMMARY")
print("="*50)
print(f"Total Complaints (Rows): {df.shape[0]}")
print(f"Total Attributes (Columns): {df.shape[1]}")

print("\n--- Available Columns ---")
print(df.columns.tolist())

print("\n--- Sample Complaints & Locations ---")
# Check karte hain ki description aur location columns kaise hain
print(df[['issue_description', 'location_street_address', 'municipal_department']].head(3))
print("="*50)