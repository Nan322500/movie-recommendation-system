import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

# Load filtered movies
@st.cache_data
def load_movies():
    return pd.read_csv("filtered_movies.csv")

# Load vectorizer and indices
@st.cache_resource
def load_resources():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("movie_indices.pkl", "rb") as f:
        indices = pickle.load(f)
    indices = {k.strip().lower(): v for k, v in indices.items()}
    return tfidf, indices

movies = load_movies()
tfidf, indices = load_resources()

movie_titles = movies["title"].tolist()

def get_recommendations(title, top_n=5):
    title = title.strip().lower()
    if title not in indices:
        st.error(f"Movie '{title}' not found in the database.")
        return []

    idx = indices[title]

    # Vectorize all movie descriptions
    tfidf_matrix = tfidf.transform(movies['description'].fillna(''))

    # Compute cosine similarity for the selected movie vector against all movies
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get indices of top_n similar movies (excluding the selected one)
    sim_indices = cosine_sim.argsort()[-(top_n + 1):][::-1]
    sim_indices = sim_indices[sim_indices != idx]  # remove the selected movie itself

    return movies["title"].iloc[sim_indices].tolist()

selected_movie = st.selectbox("Choose a movie you like:", sorted(movie_titles))

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        st.subheader("📽️ Recommended Movies")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("No recommendations found. Please try another movie.")
