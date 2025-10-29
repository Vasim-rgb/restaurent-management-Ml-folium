
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import os
import json
from lightgbm import LGBMClassifier
from scipy.sparse import hstack



file_path = "Dataset .csv"  # path to your uploaded file
min_samples = 100           # keep classes with >= min_samples examples
test_size = 0.20
random_state = 42
# ---------------

# 1. Load data
df = pd.read_csv(file_path)
print("Original shape:", df.shape)

# 2. Drop rows with missing Cuisines and create single-label target
df = df.dropna(subset=["Cuisines"]).copy()
df["Primary_Cuisine"] = df["Cuisines"].apply(lambda x: str(x).split(",")[0].strip())

# 3. Filter rare classes (reduce label explosion)
counts = df["Primary_Cuisine"].value_counts()
popular = counts[counts >= min_samples].index.tolist()
df = df[df["Primary_Cuisine"].isin(popular)].copy()
print(f"After keeping cuisines with >= {min_samples} samples:", df.shape)
print("Kept cuisines:", df["Primary_Cuisine"].nunique())

# 4. Select base features
keep_cols = ["City", "Locality", "Average Cost for two", "Currency", "Has Table booking",
             "Has Online delivery", "Is delivering now", "Price range", "Aggregate rating",
             "Votes", "Rating text"]
keep_cols = [c for c in keep_cols if c in df.columns]
X = df[keep_cols].copy()
y = df["Primary_Cuisine"].copy()

# 5. Label encode target (early to use for stratification and model training)
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print("Classes:", classes)


# --- Advanced Feature Engineering ---

# Create new features from the 'Cuisines' column
df_extended = df.copy()
df_extended["num_cuisines"] = df_extended["Cuisines"].apply(
    lambda x: len(str(x).split(","))
)
def get_cuisine_at_index(cuisine_str, index):
    cuisines = [c.strip() for c in str(cuisine_str).split(",")]
    if len(cuisines) > index:
        return cuisines[index]
    return "None"
df_extended["second_cuisine"] = df_extended["Cuisines"].apply(lambda x: get_cuisine_at_index(x, 1))
df_extended["third_cuisine"] = df_extended["Cuisines"].apply(lambda x: get_cuisine_at_index(x, 2))

# Create geographic features
df_extended["city_len"] = df_extended["City"].apply(lambda x: len(str(x)))
df_extended["locality_len"] = df_extended["Locality"].apply(lambda x: len(str(x)))

# Create features from 'Rating text'
df_extended["rating_text_word_count"] = df_extended["Rating text"].apply(lambda x: len(str(x).split()))

# Create interaction terms
if "Average Cost for two" in df_extended.columns and "Price range" in df_extended.columns:
    df_extended["cost_x_pricerange"] = df_extended["Average Cost for two"] * df_extended["Price range"]
if "Aggregate rating" in df_extended.columns and "Votes" in df_extended.columns:
    df_extended["rating_x_votes"] = df_extended["Aggregate rating"] * df_extended["Votes"]

# Update keep_cols_extended to include all features now
keep_cols_extended = list(keep_cols) + ["num_cuisines", "second_cuisine", "third_cuisine",
                                        "city_len", "locality_len", "rating_text_word_count"]
if "cost_x_pricerange" in df_extended.columns: keep_cols_extended.append("cost_x_pricerange")
if "rating_x_votes" in df_extended.columns: keep_cols_extended.append("rating_x_votes")

# Filter keep_cols_extended to only include columns actually present
keep_cols_extended = [c for c in keep_cols_extended if c in df_extended.columns]
X_extended = df_extended[keep_cols_extended].copy()

# --- Explore Text Data (TF-IDF) ---
from sklearn.feature_extraction.text import TfidfVectorizer

df_extended["combined_text"] = (
    df_extended["Restaurant Name"].astype(str).fillna("") + " " +
    df_extended["Cuisines"].astype(str).fillna("")
)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_text = tfidf.fit_transform(df_extended["combined_text"])
print("Shape of TF-IDF features:", X_text.shape)


# --- Re-process the extended feature DataFrame and combine it with text features ---

numeric_cols_extended = X_extended.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_extended = [c for c in X_extended.columns if c not in numeric_cols_extended]
print("Numeric columns in X_extended:", numeric_cols_extended)
print("Categorical columns in X_extended:", cat_cols_extended)

# Refit and apply imputation and scaling to numeric columns in X_extended
num_imputer_extended = SimpleImputer(strategy="median")
X_num_processed = num_imputer_extended.fit_transform(X_extended[numeric_cols_extended])
scaler_extended = StandardScaler()
X_num_processed = scaler_extended.fit_transform(X_num_processed)

# Refit and apply imputation and ordinal encoding to categorical columns in X_extended
cat_imputer_extended = SimpleImputer(strategy="constant", fill_value="NA")
X_cat_imputed = cat_imputer_extended.fit_transform(X_extended[cat_cols_extended])
ord_enc_extended = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat_processed = ord_enc_extended.fit_transform(X_cat_imputed)

# Horizontally stack processed features and text features
X_proc_extended = hstack([X_num_processed, X_cat_processed, X_text])
print("Shape of X_proc_extended:", X_proc_extended.shape)

# 8. Train/test split (stratify) on the extended data
X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(
    X_proc_extended, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
)
print("Train/test split performed on X_proc_extended and y_enc.")


# --- Train LightGBM with Best Parameters ---

# --- IMPORTANT: Replace with the best parameters found from your RandomizedSearchCV ---
# Example placeholder parameters - replace with your actual best parameters
best_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
# ----------------------------------------------------------------------------------

print("\nTraining LightGBM model with best parameters...")
tuned_lgbm = LGBMClassifier(objective='multiclass', num_class=len(classes),
                             random_state=random_state, **best_params)

tuned_lgbm.fit(X_train_adv, y_train_adv)
print("LightGBM training finished.")


# --- Evaluation of Tuned LightGBM Model ---
print("\nEvaluating the best tuned LightGBM model on the test set...")
y_pred_tuned = tuned_lgbm.predict(X_test_adv)

acc_tuned = accuracy_score(y_test_adv, y_pred_tuned)
prec_tuned = precision_score(y_test_adv, y_pred_tuned, average="weighted", zero_division=0)
rec_tuned = recall_score(y_test_adv, y_pred_tuned, average="weighted", zero_division=0)
f1_tuned = f1_score(y_test_adv, y_pred_tuned, average="weighted", zero_division=0)

print(f"Tuned LightGBM - Acc: {acc_tuned:.4f}, Prec: {prec_tuned:.4f}, Rec: {rec_tuned:.4f}, F1: {f1_tuned:.4f}")
print("Classification report (Tuned LightGBM):\n", classification_report(y_test_adv, y_pred_tuned, target_names=classes, zero_division=0))

# Confusion matrix for the tuned LightGBM
cm_tuned_lgbm = confusion_matrix(y_test_adv, y_pred_tuned)
disp_tuned_lgbm = ConfusionMatrixDisplay(confusion_matrix=cm_tuned_lgbm, display_labels=classes)
fig_tuned_lgbm, ax_tuned_lgbm = plt.subplots(figsize=(12,12))
disp_tuned_lgbm.plot(ax=ax_tuned_lgbm, xticks_rotation=90)
plt.title("Confusion Matrix - Tuned LightGBM Model")
plt.tight_layout()
plt.show()


# --- Save the final tuned LightGBM model and preprocessing objects ---

# Save the trained model and preprocessing objects
model_artifacts = {
    "tuned_lgbm": tuned_lgbm,
    "label_encoder": le, # Trained on the full filtered data
    "ordinal_encoder_extended": ord_enc_extended, # Trained on X_extended categorical
    "num_imputer_extended": num_imputer_extended, # Trained on X_extended numeric
    "scaler_extended": scaler_extended,         # Trained on X_extended numeric
    "tfidf_vectorizer": tfidf,                  # Trained on combined text
    "feature_columns_extended": keep_cols_extended, # List of feature names in X_extended
    "numeric_cols_extended": numeric_cols_extended,
    "cat_cols_extended": cat_cols_extended,
    "classes": classes # List of class names
}

joblib.dump(model_artifacts, "cuisine_lightgbm_model.joblib")
print("\nSaved final tuned LightGBM model and preprocessing objects to cuisine_lightgbm_model.joblib")

# Compute and save metrics
metrics = {
    "Original Dataset Shape": df.shape,
    "Filtered Dataset Shape": df_extended.shape,
    "Number of Unique Cuisines": df["Primary_Cuisine"].nunique(),
    "Number of Classes": len(classes),
    "Test Size": test_size,
    "Accuracy": acc_tuned,
    "Precision": prec_tuned,
    "Recall": rec_tuned,
    "F1 Score": f1_tuned,
    "Number of Numeric Features": len(numeric_cols_extended),
    "Number of Categorical Features": len(cat_cols_extended),
    "TF-IDF Max Features": tfidf.max_features,
    "Processed Features Shape": X_proc_extended.shape,
    "Train Set Size": X_train_adv.shape[0],
    "Test Set Size": X_test_adv.shape[0],
    "Model Parameters": best_params
}
os.makedirs("TASK_3", exist_ok=True)
with open("TASK_3/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("✅ Metrics saved successfully to TASK_3/metrics.json!")

print("\nEnd of Combined Code (LightGBM with Best Params).")
