import streamlit as st
import joblib
import pandas as pd
from fuzzywuzzy import process

# Set the page title and layout
st.set_page_config(page_title="Music Recommender", layout="wide", page_icon="🎵")

# ---- Load the Models and Data ----
# We use st.cache_resource so it only loads the heavy files ONCE, making the site super fast.
@st.cache_resource
def load_assets():
    model = joblib.load("bollywood_knn.pkl")
    data = pd.read_csv("bollywood_data.csv")
    features = joblib.load("bollywood_features.pkl")
    return model, data, features

model, music_data, feature_columns = load_assets()

# Create two columns for the wide layout: Left for inputs/info, Right for recommendations
col_left, col_right = st.columns([1, 1.2], gap="large")

# Pre-declare user song search variable
user_song = ""

with col_left:
    st.title("🎵 AI Music Recommendation System")
    st.write("Type in a song you like, and the AI will use **K-Nearest Neighbors** to find 5 songs with the exact same musical vibe!")
    
    # ---- User Input ----
    # This creates a search box on the screen
    search_input = st.text_input("Search for a song:", placeholder="e.g., Shape of You")
    
    # Quick suggestion chips for a clean "student project" interaction
    st.write("💡 *Quick try one of these popular songs:*")
    suggestions = ["Badtameez Dil", "Tum Hi Ho", "Kabira"]
    cols = st.columns(len(suggestions))
    
    selected_suggestion = None
    for idx, song in enumerate(suggestions):
        if cols[idx].button(song, use_container_width=True):
            selected_suggestion = song
            
    # Set user_song to suggestion if clicked, else the input box text
    user_song = selected_suggestion if selected_suggestion else search_input

# ---- Recommendation Logic ----
matched_song = None
score = None
recommendations_sorted = None

if user_song:
    # 1. Fuzzy matching (Finds the closest spelling match)
    all_song_names = music_data['track_name'].unique()
    matched_song, score = process.extractOne(user_song, all_song_names)
    
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
    recommendations_sorted = recommendations.sort_values('similarity_score (%)', ascending=False)

# ---- Render Left Column Output ----
with col_left:
    if user_song and matched_song:
        st.write("---")
        # Clean display for the selected song using Streamlit columns & metric
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"**Selected Song:**\n### 🎵 {matched_song}")
        with col_b:
            st.metric(label="Fuzzy Match Score", value=f"{score}/100")
        
        if score < 60:
            st.warning("That song might not be in our database. Here is the closest match we found!")
            
    # ---- About Section (Student Project Style) ----
    st.write("---")
    with st.expander("ℹ️ About This Project (How it Works)"):
        st.markdown("""
        This is a **K-Nearest Neighbors (KNN)** based music recommendation system.
        
        **How the model works:**
        1. **Fuzzy Matching:** When you search a song, the system uses fuzzy string matching to find the closest match in our Bollywood database.
        2. **Feature Extraction:** It retrieves the musical features of the selected song (acousticness, danceability, energy, loudness, tempo, popularity, etc.).
        3. **KNN Search:** The KNN model measures the distance between the song's features and all other songs in the database.
        4. **Recommendation:** The 5 closest songs with the shortest distance are recommended as sharing a similar vibe!
        
        *Built using Streamlit, Scikit-Learn, Pandas, and Joblib.*
        """)

# ---- Render Right Column Output ----
with col_right:
    if not user_song:
        st.info("👈 Enter a song or click a suggestion on the left to get started!")
    else:
        st.subheader("🔥 Top 5 Recommendations:")
        
        # Display recommendations as clean rows with metrics inside a border container
        for idx, row in recommendations_sorted.iterrows():
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{row['track_name']}**")
                    st.caption(f"👤 {row['artist_name']} | 🏷️ {row['genre'].title()}")
                with col2:
                    # Popularity score is represented as a percentage out of 100
                    pop_percent = int(row['popularity'] * 100) if row['popularity'] <= 1 else int(row['popularity'])
                    st.metric(label="Popularity", value=f"{pop_percent}%")
                with col3:
                    st.metric(label="Match Vibe", value=f"{row['similarity_score (%)']:.1f}%")
        
        st.success("Done! Feel free to search for another song.")
