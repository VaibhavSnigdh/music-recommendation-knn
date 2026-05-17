import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

print("Loading data...")
music_data = pd.read_csv("spotify_data.csv")
music_data.dropna(inplace=True)

print("Filtering for ONLY Indian/Bollywood songs...")
# Filter for Indian genres
indian_genres = ['pop-film', 'indian', 'desi', 'punjabi', 'hindi', 'tamil', 'telugu']
music_data = music_data[music_data['genre'].str.lower().isin(indian_genres)]

# Keep relatively popular songs to ensure good recommendations
music_data = music_data[music_data['popularity'] >= 20]

if 'Unnamed: 0' in music_data.columns:
    music_data.drop(columns=['Unnamed: 0'], inplace=True)
if 'track_id' in music_data.columns:
    music_data.drop(columns=['track_id'], inplace=True)

print(f"Total Indian songs found: {len(music_data)}")

print("Encoding categorical features...")
label_encoder = LabelEncoder()
categorical_cols = ['artist_name', 'track_name', 'genre', 'year']

for col in categorical_cols:
    music_data[col + '_encoded'] = label_encoder.fit_transform(music_data[col])

print("Normalizing numerical features...")
def normalize_column(col):
    max_val = music_data[col].max()
    min_val = music_data[col].min()
    if max_val != min_val:
        music_data[col] = (music_data[col] - min_val) / (max_val - min_val)
    else:
        music_data[col] = 0

numerical_cols = music_data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
for col in numerical_cols:
    if col != 'year':
        normalize_column(col)

print("Preparing features...")
# To fix the alphabetical bias issue, we will drop the encoded text columns from the KNN features!
# The AI should recommend based on audio features (danceability, energy, acousticness) 
# NOT the alphabetical spelling of the song name.
X_cluster = music_data.select_dtypes(np.number).drop(columns=['artist_name_encoded', 'track_name_encoded'])

print("Clustering...")
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=10, random_state=42))
])
music_data['cluster'] = cluster_pipeline.fit_predict(X_cluster)

X = X_cluster.drop(columns=['cluster', 'year'], errors='ignore').copy()
y = music_data['cluster']

print("Training KNN Model...")
model = KNeighborsClassifier(n_neighbors=4, metric='cosine', algorithm='brute')
model.fit(X, y)

print("Saving models to disk...")
joblib.dump(model, "bollywood_knn.pkl")

columns_to_keep = ['track_name', 'artist_name', 'genre', 'popularity'] + list(X.columns)
app_data = music_data[columns_to_keep]

app_data.to_csv("bollywood_data.csv", index=False)
joblib.dump(list(X.columns), "bollywood_features.pkl")

print("Export complete! Files saved: bollywood_knn.pkl, bollywood_data.csv, bollywood_features.pkl")
