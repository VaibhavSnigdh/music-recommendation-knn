import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

print("Loading data...")
music_data = pd.read_csv("spotify_data.csv")
music_data.dropna(inplace=True)
music_data = music_data[music_data['popularity'] >= 40]

if 'Unnamed: 0' in music_data.columns:
    music_data.drop(columns=['Unnamed: 0'], inplace=True)
if 'track_id' in music_data.columns:
    music_data.drop(columns=['track_id'], inplace=True)

print("Encoding categorical features...")
label_encoder = LabelEncoder()
categorical_cols = ['artist_name', 'track_name', 'genre', 'year']

for col in categorical_cols:
    music_data[col + '_encoded'] = label_encoder.fit_transform(music_data[col])

print("Normalizing numerical features...")
def normalize_column(col):
    max_val = music_data[col].max()
    min_val = music_data[col].min()
    music_data[col] = (music_data[col] - min_val) / (max_val - min_val)

numerical_cols = music_data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
for col in numerical_cols:
    if col != 'year' and not col.endswith('_encoded') and col != 'popularity':
        # wait, in notebook they normalized popularity too? Yes, "except year"
        pass
    if col != 'year':
        normalize_column(col)

print("Preparing features...")
# To match the notebook: 
# X = music_data.select_dtypes(np.number).drop(columns=['cluster', 'year']).copy()
# Wait, they added 'cluster' using KMeans. Do I need 'cluster' to train KNN?
# Let's add KMeans just like the notebook!
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

print("Clustering...")
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=10, random_state=42))
])
X_cluster = music_data.select_dtypes(np.number)
music_data['cluster'] = cluster_pipeline.fit_predict(X_cluster)

X = music_data.select_dtypes(np.number).drop(columns=['cluster', 'year']).copy()
y = music_data['cluster']

print("Training KNN Model...")
# Train the k=4 model on ALL data for the final product
model = KNeighborsClassifier(n_neighbors=4, metric='cosine', algorithm='brute')
model.fit(X, y)

print("Saving models to disk...")
# Save the model
joblib.dump(model, "knn_model.pkl")

# Save the dataset (only keeping columns we actually need for the app to save space!)
# We need track_name, artist_name, genre, popularity, and all features in X
columns_to_keep = ['track_name', 'artist_name', 'genre', 'popularity'] + list(X.columns)
app_data = music_data[columns_to_keep]

# Save the app data as CSV to avoid version mismatch issues
app_data.to_csv("music_data.csv", index=False)
# Also save the list of feature columns
joblib.dump(list(X.columns), "feature_columns.pkl")

print("Export complete! Files saved: knn_model.pkl, music_data.csv, feature_columns.pkl")
