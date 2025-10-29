# ----------------------------------------------------------
# TASK-2: Restaurant Recommendation System - Model Training
# ----------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import json

# Step 1: Load Dataset
file_path = "Dataset .csv"  # Adjust if needed
data = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully!")

# Step 2: Data Preprocessing
cols_to_drop = ['Restaurant ID', 'Address', 'Locality Verbose', 'Rating color', 'Rating text']
df = data.drop(columns=cols_to_drop)
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

label_enc = LabelEncoder()
for col in ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Currency', 'City']:
    df[col] = label_enc.fit_transform(df[col])

# Step 3: Combine features for similarity
df['combined_features'] = (
    df['Cuisines'].astype(str) + ' ' +
    df['City'].astype(str) + ' ' +
    df['Price range'].astype(str)
)

# Step 4: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Step 5: Cosine Similarity (optional to save space)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 6: Save all necessary components
os.makedirs("TASK_2", exist_ok=True)
joblib.dump(df, "TASK_2/restaurant_data.pkl")
joblib.dump(tfidf, "TASK_2/tfidf_vectorizer.pkl")
print("✅ TF-IDF vectorizer & data saved successfully in TASK-2 folder!")

# Step 7: Compute and save metrics
metrics = {
    "Number of Restaurants": len(df),
    "Number of Unique Cuisines": df['Cuisines'].nunique(),
    "Average Price Range": df['Price range'].mean(),
    "Number of TF-IDF Features": tfidf_matrix.shape[1],
    "Mean Cosine Similarity": cosine_sim.mean(),
    "Total Cities": df['City'].nunique(),
    "Average Rating": df['Aggregate rating'].mean(),
    "Percentage with Online Delivery": (df['Has Online delivery'].sum() / len(df)) * 100
}
with open("TASK_2/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("✅ Metrics saved successfully to TASK-2/metrics.json!")
