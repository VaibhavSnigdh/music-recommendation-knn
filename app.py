import streamlit as st
import joblib
import pandas as pd
from fuzzywuzzy import process

# Set the page title and layout
st.set_page_config(page_title="Music Recommender", layout="centered", page_icon="🎵")

# ---- Title and Description ----
st.title("🎵 AI Music Recommendation System")
st.write("Type in a song you like, and the AI will use **K-Nearest Neighbors** to find 5 songs with the exact same musical vibe!")

# ---- Load the Models and Data ----
# We use st.cache_resource so it only loads the heavy files ONCE, making the site super fast.
@st.cache_resource
def load_assets():
    model = joblib.load("bollywood_knn.pkl")
    data = pd.read_csv("bollywood_data.csv")
    features = joblib.load("bollywood_features.pkl")
    return model, data, features

model, music_data, feature_columns = load_assets()

# ---- User Input ----
# This creates a search box on the screen
user_song = st.text_input("Search for a song:", placeholder="e.g., Shape of You")

# ---- Recommendation Logic ----
if user_song:
    st.write("---")
    # 1. Fuzzy matching (Finds the closest spelling match)
    all_song_names = music_data['track_name'].unique()
    matched_song, score = process.extractOne(user_song, all_song_names)
    
    st.write(f"**Selected Song:** {matched_song} *(Match Score: {score}/100)*")
    
    if score < 60:
        st.warning("That song might not be in our database. Here is the closest match we found!")
    
    # 2. Get the musical features for the matching song
    matching_songs = music_data[music_data['track_name'] == matched_song]
    
    # If there are multiple versions of the song, pick the most popular one
    if len(matching_songs) > 1:
        input_song_data = matching_songs.sort_values('popularity', ascending=False).head(1)
    else:
        input_song_data = matching_songs
        
    input_features = input_song_data[feature_columns]
    
    # 3. Ask the KNN model to find the 5 closest neighbors (plus 1 for the original song)
    distances, indices = model.kneighbors(input_features, n_neighbors=6)
    
    # 4. Extract the top 5 recommendations
    recommended_indices = indices[0][1:] # Skip the 1st one because it is the song itself
    recommendations = music_data.iloc[recommended_indices][['track_name', 'artist_name', 'genre', 'popularity']].copy()
    
    # Add a similarity score (converting distance to a percentage out of 100)
    recommendations['similarity_score (%)'] = (1 - distances[0][1:]) * 100
    recommendations['similarity_score (%)'] = recommendations['similarity_score (%)'].round(2)
    
    # 5. Display the results on the website nicely
    st.subheader("🔥 Top 5 Recommendations:")
    
    # Display as a clean dataframe without the messy index numbers
    st.dataframe(
        recommendations.sort_values('similarity_score (%)', ascending=False),
        hide_index=True,
        use_container_width=True
    )
    
    st.success("Done! Feel free to search for another song.")
