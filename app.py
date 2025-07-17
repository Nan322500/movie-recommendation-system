import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")

# Load files
@st.cache_resource
def load_data():
    # Load filtered movies
    movies = pd.read_csv("filtered_movies.csv")

    # Load cosine similarity matrix
    cosine_sim = np.load("cosine_similarity.npy")

    # Load vectorizer (not used in app but useful if needed later)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # Load movie title → index mapping
    with open("movie_indices.pkl", "rb") as f:
        indices = pickle.load(f)

    return movies, cosine_sim, indices

movies, cosine_sim, indices = load_data()

# Clean the indices keys just in case
indices = {k.strip().lower(): v for k, v in indices.items()}
movie_titles = movies["title"].tolist()

# Recommend function
def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.strip().lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()

# Streamlit UI
selected_movie = st.selectbox("Choose a movie you like:", sorted(movie_titles))

if st.button("Recommend"):
    st.subheader("📽️ Recommended Movies")
    results = get_recommendations(selected_movie)
    if results:
        for i, rec in enumerate(results, 1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("No recommendations found. Please try another movie.")
