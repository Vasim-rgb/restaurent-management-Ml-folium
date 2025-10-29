import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os
import json

# --------------------------------------------
# STEP 1: Ensure Required Folders Exist
# --------------------------------------------
os.makedirs("TASK_4/static", exist_ok=True)
os.makedirs("TASK_4/templates", exist_ok=True)

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
data['Aggregate rating'] = data['Aggregate rating'].fillna(data['Aggregate rating'].mode()[0])
data['Price range'] = data['Price range'].fillna(data['Price range'].mode()[0])

# --------------------------------------------
# STEP 4: Group Data by City and Locality for Concentration Analysis
# --------------------------------------------
city_stats = data.groupby('City').agg({
    'Restaurant Name': 'count',
    'Aggregate rating': 'mean',
    'Price range': 'mean',
    'Average Cost for two': 'mean',
    'Votes': 'sum'
}).rename(columns={'Restaurant Name': 'Restaurant Count'})

# Group by Locality for more granular analysis
locality_stats = data.groupby('Locality Verbose').agg({
    'Restaurant Name': 'count',
    'Aggregate rating': 'mean',
    'Price range': 'mean'
}).rename(columns={'Restaurant Name': 'Restaurant Count'})

# Find most common cuisine per city
def get_most_common_cuisine(series):
    if len(series) == 0:
        return 'Unknown'
    cuisines = series.str.split(',').explode().str.strip().value_counts()
    return cuisines.index[0] if len(cuisines) > 0 else 'Unknown'

city_cuisines = data.groupby('City')['Cuisines'].apply(get_most_common_cuisine)

# Find most common cuisine per locality
locality_cuisines = data.groupby('Locality Verbose')['Cuisines'].apply(get_most_common_cuisine)

top_cities = city_stats.sort_values('Restaurant Count', ascending=False).head(10)
top_rated = city_stats.sort_values('Aggregate rating', ascending=False).head(10)
top_localities = locality_stats.sort_values('Restaurant Count', ascending=False).head(10)

# --------------------------------------------
# STEP 5: Plot Visualizations
# --------------------------------------------

# Top Cities by Count
plt.figure(figsize=(10,6))
sns.barplot(x=top_cities.index, y=top_cities['Restaurant Count'], palette='Blues_d')
plt.title("Top 10 Cities with Most Restaurants")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("TASK_4/static/top_cities.png")
plt.close()

# Top Cities by Rating
plt.figure(figsize=(10,6))
sns.barplot(x=top_rated.index, y=top_rated['Aggregate rating'], palette='Greens_d')
plt.title("Top 10 Cities by Average Rating")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig("TASK_4/static/top_rated.png")
plt.close()

# Top Localities by Restaurant Count
plt.figure(figsize=(12,6))
sns.barplot(x=top_localities.index, y=top_localities['Restaurant Count'], palette='Oranges_d')
plt.title("Top 10 Localities with Most Restaurants")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("TASK_4/static/top_localities.png")
plt.close()

# Rating Distribution by City
plt.figure(figsize=(12,8))
top_10_cities = city_stats.sort_values('Restaurant Count', ascending=False).head(10)
sns.boxplot(data=data[data['City'].isin(top_10_cities.index)], x='City', y='Aggregate rating', palette='Set3')
plt.title("Rating Distribution in Top 10 Cities")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Aggregate Rating")
plt.tight_layout()
plt.savefig("TASK_4/static/rating_distribution.png")
plt.close()

# Price Range Distribution by City
plt.figure(figsize=(12,8))
sns.boxplot(data=data[data['City'].isin(top_10_cities.index)], x='City', y='Price range', palette='Set2')
plt.title("Price Range Distribution in Top 10 Cities")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Price Range")
plt.tight_layout()
plt.savefig("TASK_4/static/price_distribution.png")
plt.close()

print("✅ Saved plots: static/top_cities.png, static/top_rated.png, static/top_localities.png, static/rating_distribution.png, static/price_distribution.png")

# --------------------------------------------
# STEP 6: Create Interactive Map with Clustering Analysis
# --------------------------------------------
center_lat, center_long = data['Latitude'].mean(), data['Longitude'].mean()
m = folium.Map(location=[center_lat, center_long], zoom_start=2)

# Create different colored markers based on rating ranges
def get_color(rating):
    if rating >= 4.0:
        return 'green'
    elif rating >= 3.0:
        return 'orange'
    elif rating >= 2.0:
        return 'blue'
    else:
        return 'red'

# Sample restaurants for map (limit to avoid performance issues)
sample_data = data.sample(min(1000, len(data)))

for _, row in sample_data.iterrows():
    color = get_color(row['Aggregate rating'])
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
    popup=f"<b>{row['Restaurant Name']}</b><br>Rating: {row['Aggregate rating']}<br>City: {row['City']}<br>Cuisine: {str(row['Cuisines'])[:50]}..."
    ).add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed;
     bottom: 50px; left: 50px; width: 150px; height: 120px;
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
     border-radius:6px; padding: 10px">
     <p><b>Rating Legend</b></p>
     <p><span style="color:green;">●</span> 4.0+</p>
     <p><span style="color:orange;">●</span> 3.0-3.9</p>
     <p><span style="color:blue;">●</span> 2.0-2.9</p>
     <p><span style="color:red;">●</span> <2.0</p>
</div>
'''

m.get_root().html.add_child(folium.Element(legend_html))

m.save("TASK_4/templates/restaurants_map.html")
print("✅ Saved interactive map: templates/restaurants_map.html")

# --------------------------------------------
# STEP 7: Compute and Save Comprehensive Metrics
# --------------------------------------------
# Calculate cuisine diversity per city
city_cuisine_diversity = data.groupby('City')['Cuisines'].apply(lambda x: x.str.split(',').explode().str.strip().nunique())

# Calculate restaurant density (restaurants per square km approximation)
# Using rough estimation based on city population data patterns
city_density = city_stats['Restaurant Count'] / city_stats['Restaurant Count'].max()  # Normalized density

# Calculate rating variance per city
city_rating_variance = data.groupby('City')['Aggregate rating'].var()

# Calculate price range variance per city
city_price_variance = data.groupby('City')['Price range'].var()

# Find cities with highest concentration of high-rated restaurants
high_rated_threshold = data['Aggregate rating'].quantile(0.75)
high_rated_concentration = data[data['Aggregate rating'] >= high_rated_threshold].groupby('City').size() / data.groupby('City').size()

# Calculate geographical spread (variance in coordinates)
city_lat_variance = data.groupby('City')['Latitude'].var()
city_long_variance = data.groupby('City')['Longitude'].var()

# Additional Insights: Country-wise analysis
country_stats = data.groupby('Country Code').agg({
    'Restaurant Name': 'count',
    'Aggregate rating': 'mean',
    'Price range': 'mean',
    'Average Cost for two': 'mean',
    'Votes': 'sum'
}).rename(columns={'Restaurant Name': 'Restaurant Count'})

# Most common cuisine by country
country_cuisines = data.groupby('Country Code')['Cuisines'].apply(get_most_common_cuisine)

# Rating distribution percentiles
rating_percentiles = data['Aggregate rating'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])

# Price range vs rating correlation
price_rating_corr = data['Price range'].corr(data['Aggregate rating'])

# Top restaurants by votes
top_voted_restaurants = data.nlargest(5, 'Votes')[['Restaurant Name', 'City', 'Votes', 'Aggregate rating']]

# Service availability by country
country_services = data.groupby('Country Code').agg({
    'Has Table booking': lambda x: (x == 'Yes').mean() * 100,
    'Has Online delivery': lambda x: (x == 'Yes').mean() * 100,
    'Is delivering now': lambda x: (x == 'Yes').mean() * 100
})

# Cuisine diversity by country
country_cuisine_diversity = data.groupby('Country Code')['Cuisines'].apply(lambda x: x.str.split(',').explode().str.strip().nunique())

# Cities with highest/lowest average ratings
highest_rated_cities = city_stats.nlargest(5, 'Aggregate rating')[['Aggregate rating', 'Restaurant Count']]
lowest_rated_cities = city_stats.nsmallest(5, 'Aggregate rating')[['Aggregate rating', 'Restaurant Count']]

# Price range distribution analysis
price_range_counts = data['Price range'].value_counts().sort_index()

# Rating color distribution
rating_color_counts = data['Rating color'].value_counts()

# Geographical clustering insights (simple distance-based)
# Calculate average distance from city center (simplified)
city_centers = data.groupby('City')[['Latitude', 'Longitude']].mean()
city_spread = data.groupby('City').apply(lambda x: ((x['Latitude'] - city_centers.loc[x.name, 'Latitude'])**2 + (x['Longitude'] - city_centers.loc[x.name, 'Longitude'])**2)**0.5).mean()

metrics = {
    # Basic Statistics
    "total_restaurants": int(len(data)),
    "unique_cities": int(data['City'].nunique()),
    "unique_countries": int(data['Country Code'].nunique()),
    "unique_cuisines": int(data['Cuisines'].str.split(',').explode().str.strip().nunique()),

    # Rating Statistics
    "average_rating": float(data['Aggregate rating'].mean()),
    "median_rating": float(data['Aggregate rating'].median()),
    "min_rating": float(data['Aggregate rating'].min()),
    "max_rating": float(data['Aggregate rating'].max()),
    "rating_std_dev": float(data['Aggregate rating'].std()),
    "rating_25th_percentile": float(rating_percentiles[0.25]),
    "rating_75th_percentile": float(rating_percentiles[0.75]),
    "rating_90th_percentile": float(rating_percentiles[0.9]),
    "rating_95th_percentile": float(rating_percentiles[0.95]),

    # Price Statistics
    "average_price_range": float(data['Price range'].mean()),
    "median_price_range": float(data['Price range'].median()),
    "average_cost_for_two": float(data['Average Cost for two'].mean()),
    "median_cost_for_two": float(data['Average Cost for two'].median()),
    "price_rating_correlation": float(price_rating_corr),

    # Engagement Statistics
    "total_votes": int(data['Votes'].sum()),
    "average_votes": float(data['Votes'].mean()),
    "percentage_with_table_booking": float((data['Has Table booking'] == 'Yes').mean() * 100),
    "percentage_with_online_delivery": float((data['Has Online delivery'] == 'Yes').mean() * 100),
    "percentage_delivering_now": float((data['Is delivering now'] == 'Yes').mean() * 100),

    # Top Performers
    "top_city_by_count": str(top_cities.index[0]),
    "top_city_count": int(top_cities['Restaurant Count'].iloc[0]),
    "top_city_by_rating": str(top_rated.index[0]),
    "top_city_rating": float(top_rated['Aggregate rating'].iloc[0]),
    "top_locality_by_count": str(top_localities.index[0]),
    "top_locality_count": int(top_localities['Restaurant Count'].iloc[0]),

    # Cuisine Analysis
    "most_common_cuisine": str(data['Cuisines'].str.split(',').explode().str.strip().value_counts().index[0]),
    "average_cuisine_diversity_per_city": float(city_cuisine_diversity.mean()),
    "average_cuisine_diversity_per_country": float(country_cuisine_diversity.mean()),

    # Geographical Insights
    "map_center_lat": float(center_lat),
    "map_center_long": float(center_long),
    "sampled_points_on_map": int(min(1000, len(data))),
    "latitude_range": float(data['Latitude'].max() - data['Latitude'].min()),
    "longitude_range": float(data['Longitude'].max() - data['Longitude'].min()),
    "average_city_spread": float(city_spread.mean()),

    # Concentration Analysis
    "cities_with_high_rated_restaurants": int(high_rated_concentration.count()),
    "average_high_rated_concentration": float(high_rated_concentration.mean() * 100),
    "most_concentrated_city": str(high_rated_concentration.idxmax()) if not high_rated_concentration.empty else "N/A",
    "highest_concentration_percentage": float(high_rated_concentration.max() * 100) if not high_rated_concentration.empty else 0.0,

    # Variance Analysis (indicating diversity/spread)
    "average_rating_variance_across_cities": float(city_rating_variance.mean()),
    "average_price_variance_across_cities": float(city_price_variance.mean()),
    "average_geographical_spread_lat": float(city_lat_variance.mean()),
    "average_geographical_spread_long": float(city_long_variance.mean()),

    # Country-wise Insights
    "top_country_by_restaurants": str(country_stats['Restaurant Count'].idxmax()),
    "top_country_restaurant_count": int(country_stats['Restaurant Count'].max()),
    "countries_with_table_booking": int((country_services['Has Table booking'] > 0).sum()),
    "countries_with_online_delivery": int((country_services['Has Online delivery'] > 0).sum()),

    # Distribution Insights
    "price_range_1_count": int(price_range_counts.get(1, 0)),
    "price_range_2_count": int(price_range_counts.get(2, 0)),
    "price_range_3_count": int(price_range_counts.get(3, 0)),
    "price_range_4_count": int(price_range_counts.get(4, 0)),
    "rating_color_distribution": rating_color_counts.to_dict(),

    # Top Voted Restaurants
    "top_voted_restaurants": top_voted_restaurants.to_dict('records'),

    # City Rating Extremes
    "highest_rated_cities": highest_rated_cities.to_dict(),
    "lowest_rated_cities": lowest_rated_cities.to_dict()
}

with open("TASK_4/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved successfully to TASK-4/metrics.json!")

# --------------------------------------------
# STEP 8: Summary Output and Insights
# --------------------------------------------
print("\n🏁 Location-based Analysis Complete!")
print("\n📊 Generated Files:")
print("├── static/top_cities.png - Top cities by restaurant count")
print("├── static/top_rated.png - Top cities by average rating")
print("├── static/top_localities.png - Top localities by restaurant count")
print("├── static/rating_distribution.png - Rating distribution across cities")
print("├── static/price_distribution.png - Price range distribution across cities")
print("└── templates/restaurants_map.html - Interactive geographical map")

print("\n🔍 Key Insights:")
print(f"• Total restaurants analyzed: {len(data)}")
print(f"• Restaurants span across {data['City'].nunique()} cities in {data['Country Code'].nunique()} countries")
print(f"• Top city by restaurant count: {top_cities.index[0]} ({top_cities['Restaurant Count'].iloc[0]} restaurants)")
print(f"• Top city by average rating: {top_rated.index[0]} ({top_rated['Aggregate rating'].iloc[0]:.2f} stars)")
print(f"• Top locality by restaurant count: {top_localities.index[0]} ({top_localities['Restaurant Count'].iloc[0]} restaurants)")
print(f"• Most common cuisine: {data['Cuisines'].str.split(',').explode().str.strip().value_counts().index[0]}")
print(f"• Average cuisine diversity per city: {city_cuisine_diversity.mean():.1f} unique cuisines")
print(f"• Cities with high-rated restaurant concentration: {high_rated_concentration.count()}")
print(f"• Most concentrated high-rated city: {high_rated_concentration.idxmax() if not high_rated_concentration.empty else 'N/A'} ({high_rated_concentration.max()*100:.1f}% high-rated)")

print("\nYou can now run:  python app.py  🚀")
