from flask import Blueprint, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
from scipy.sparse import hstack

task3_bp = Blueprint('task3', __name__, template_folder='templates', static_folder='static')

# ---------------------------
# Load model and preprocessors
# ---------------------------
artifacts = joblib.load(r"TASK_3/cuisine_lightgbm_model.joblib")
model = artifacts["tuned_lgbm"]
label_encoder = artifacts["label_encoder"]
ordinal_encoder = artifacts["ordinal_encoder_extended"]
num_imputer = artifacts["num_imputer_extended"]
scaler = artifacts["scaler_extended"]
tfidf = artifacts["tfidf_vectorizer"]
keep_cols = artifacts["feature_columns_extended"]
num_cols = artifacts["numeric_cols_extended"]
cat_cols = artifacts["cat_cols_extended"]
classes = artifacts["classes"]


# ---------------------------
# Feature Preprocessing
# ---------------------------
def preprocess(df):
    df = df.copy()
    # Derived cuisine features
    df["num_cuisines"] = df["Cuisines"].apply(lambda x: len(str(x).split(",")))

    def get_cuisine_at_index(cuisine_str, idx):
        cuisines = [c.strip() for c in str(cuisine_str).split(",")]
        return cuisines[idx] if len(cuisines) > idx else "None"

    df["second_cuisine"] = df["Cuisines"].apply(lambda x: get_cuisine_at_index(x, 1))
    df["third_cuisine"] = df["Cuisines"].apply(lambda x: get_cuisine_at_index(x, 2))
    df["city_len"] = df["City"].apply(lambda x: len(str(x)))
    df["locality_len"] = df["Locality"].apply(lambda x: len(str(x)))
    df["rating_text_word_count"] = df["Rating text"].apply(lambda x: len(str(x).split()))
    if "Average Cost for two" in df.columns and "Price range" in df.columns:
        df["cost_x_pricerange"] = df["Average Cost for two"] * df["Price range"]
    if "Aggregate rating" in df.columns and "Votes" in df.columns:
        df["rating_x_votes"] = df["Aggregate rating"] * df["Votes"]

    X_extended = df[[col for col in keep_cols if col in df.columns]].copy()
    X_num = num_imputer.transform(X_extended[num_cols])
    X_num = scaler.transform(X_num)
    X_cat = ordinal_encoder.transform(X_extended[cat_cols])
    df["combined_text"] = (
        df["Restaurant Name"].astype(str).fillna("") + " " + df["Cuisines"].astype(str).fillna("")
    )
    X_text = tfidf.transform(df["combined_text"])
    X_final = hstack([X_num, X_cat, X_text])
    return X_final


# ---------------------------
# Routes
# ---------------------------
@task3_bp.route("/")
def home():
    return render_template("index3.html")


@task3_bp.route("/predict", methods=["POST"])
def predict():
    data = request.form

    input_df = pd.DataFrame({
        "City": [data["city"]],
        "Locality": [data["locality"]],
        "Average Cost for two": [float(data["avg_cost"])],
        "Currency": [data["currency"]],
        "Has Table booking": [data["table_booking"]],
        "Has Online delivery": [data["online_delivery"]],
        "Is delivering now": [data["delivering_now"]],
        "Price range": [int(data["price_range"])],
        "Aggregate rating": [float(data["aggregate_rating"])],
        "Votes": [int(data["votes"])],
        "Rating text": [data["rating_text"]],
        "Restaurant Name": [data["restaurant_name"]],
        "Cuisines": [data["cuisines"]],
    })

    X_final = preprocess(input_df)
    pred = model.predict(X_final)
    predicted_class = label_encoder.inverse_transform(pred)[0]

    return jsonify({"prediction": predicted_class})


@task3_bp.route("/metrics")
def metrics():
    with open("TASK_3/metrics.json", "r") as f:
        metrics_data = json.load(f)
    return jsonify(metrics_data)


