from flask import Blueprint, render_template, request
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

task2_bp = Blueprint('task2', __name__, template_folder='templates', static_folder='static')

# Load saved models and data
df = joblib.load("TASK_2/restaurant_data.pkl")
tfidf = joblib.load("TASK_2/tfidf_vectorizer.pkl")
tfidf_matrix = tfidf.transform(df['combined_features'])

# Recommendation function
def recommend_restaurants(cuisine, city, price_range, top_n=5):
    user_pref = f"{cuisine} {city} {price_range}"
    user_vec = tfidf.transform([user_pref])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    indices = similarity_scores.argsort()[-top_n:][::-1]
    recs = df.iloc[indices][['Restaurant Name', 'City', 'Cuisines', 'Price range', 'Aggregate rating']]
    return recs

@task2_bp.route('/')
def home():
    return render_template('index2.html')

@task2_bp.route('/predict', methods=['POST'])
def predict():
    cuisine = request.form['cuisine']
    city = request.form['city']
    price_range = request.form['price_range']

    results = recommend_restaurants(cuisine, city, price_range, top_n=5)

    # Convert recommendations into HTML table
    table_html = results.to_html(classes='table table-bordered', index=False)

    return render_template('index2.html', prediction_table=table_html)
