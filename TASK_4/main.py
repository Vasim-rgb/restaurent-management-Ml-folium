import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os
import json

# --------------------------------------------
# STEP 1: Ensure Required Folders Exist
# --------------------------------------------
os.makedirs("TASK-4/static", exist_ok=True)
os.makedirs("TASK-4/templates", exist_ok=True)

# --------------------------------------------
# STEP 2: Load Dataset
# --------------------------------------------
file_path = "Dataset .csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("❌ Dataset.csv not found in TASK-4 folder!")

data = pd.read_csv(file_path)
print(f"✅ Dataset Loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# --------------------------------------------
# STEP 3: Data Cleaning
# --------------------------------------------
data = data.dropna(subset=['Longitude', 'Latitude'])
data['Aggregate rating'].fillna(data['Aggregate rating'].mode()[0], inplace=True)
data['Price range'].fillna(data['Price range'].mode()[0], inplace=True)

# --------------------------------------------
# STEP 4: Group Data by City
# --------------------------------------------
city_stats = data.groupby('City').agg({
    'Restaurant Name': 'count',
    'Aggregate rating': 'mean',
    'Price range': 'mean'
}).rename(columns={'Restaurant Name': 'Restaurant Count'})

top_cities = city_stats.sort_values('Restaurant Count', ascending=False).head(10)
top_rated = city_stats.sort_values('Aggregate rating', ascending=False).head(10)

# --------------------------------------------
# STEP 5: Plot Visualizations
# --------------------------------------------

# Top Cities by Count
plt.figure(figsize=(8,5))
sns.barplot(x=top_cities.index, y=top_cities['Restaurant Count'], palette='Blues_d')
plt.title("Top 10 Cities with Most Restaurants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("TASK-4/static/top_cities.png")
plt.close()

# Top Cities by Rating
plt.figure(figsize=(8,5))
sns.barplot(x=top_rated.index, y=top_rated['Aggregate rating'], palette='Greens_d')
plt.title("Top 10 Cities by Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("TASK-4/static/top_rated.png")
plt.close()

print("✅ Saved plots: static/top_cities.png and static/top_rated.png")

# --------------------------------------------
# STEP 6: Create Interactive Map
# --------------------------------------------
center_lat, center_long = data['Latitude'].mean(), data['Longitude'].mean()
m = folium.Map(location=[center_lat, center_long], zoom_start=2)

for _, row in data.sample(min(500, len(data))).iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.5
    ).add_to(m)

m.save("TASK-4/templates/restaurants_map.html")
print("✅ Saved map: templates/restaurants_map.html")

# --------------------------------------------
# STEP 7: Compute and Save Metrics
# --------------------------------------------
metrics = {
    "total_restaurants": int(len(data)),
    "unique_cities": int(data['City'].nunique()),
    "unique_countries": int(data['Country Code'].nunique()),
    "unique_cuisines": int(data['Cuisines'].str.split(',').explode().str.strip().nunique()),
    "average_rating": float(data['Aggregate rating'].mean()),
    "median_rating": float(data['Aggregate rating'].median()),
    "min_rating": float(data['Aggregate rating'].min()),
    "max_rating": float(data['Aggregate rating'].max()),
    "average_price_range": float(data['Price range'].mean()),
    "median_price_range": float(data['Price range'].median()),
    "average_cost_for_two": float(data['Average Cost for two'].mean()),
    "median_cost_for_two": float(data['Average Cost for two'].median()),
    "total_votes": int(data['Votes'].sum()),
    "average_votes": float(data['Votes'].mean()),
    "percentage_with_table_booking": float((data['Has Table booking'] == 'Yes').mean() * 100),
    "percentage_with_online_delivery": float((data['Has Online delivery'] == 'Yes').mean() * 100),
    "percentage_delivering_now": float((data['Is delivering now'] == 'Yes').mean() * 100),
    "top_city_by_count": str(top_cities.index[0]),
    "top_city_count": int(top_cities['Restaurant Count'].iloc[0]),
    "top_city_by_rating": str(top_rated.index[0]),
    "top_city_rating": float(top_rated['Aggregate rating'].iloc[0]),
    "most_common_cuisine": str(data['Cuisines'].str.split(',').explode().str.strip().value_counts().index[0]),
    "map_center_lat": float(center_lat),
    "map_center_long": float(center_long),
    "sampled_points_on_map": int(min(500, len(data)))
}

with open("TASK-4/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved successfully to TASK-4/metrics.json!")

# --------------------------------------------
# STEP 8: Summary Output
# --------------------------------------------
print("\n🏁 Files Generated Successfully!")
print("├── static/top_cities.png")
print("├── static/top_rated.png")
print("└── templates/restaurants_map.html")

print("\nYou can now run:  python app.py  🚀")