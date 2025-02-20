import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
file_path = "net.csv"
df = pd.read_csv(file_path)
df1= df.dropna()

df1['combined_features'] = df1['genres'] + ',' + df1['type']

# Create TF-IDF Matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df1['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Recommendation Function
def recommend_movie(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df1[df1['title'] == title].index[0]

    # Get a list of cosine similarities for that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar movies
    sim_scores = sim_scores[1:6]  # Get top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return df1['title'].iloc[movie_indices]

# Streamlit App
st.title(" Movie and TV Series Recommender")

# Select a Movie Title for Recommendation
selected_title = st.selectbox(
    "Select a Movie or TV Series for Recommendations",
    df1['title'].values
)

if st.button("Get Recommendations"):
    recommendations = recommend_movie(selected_title)
    st.subheader("Top 5 Recommendations:")
    for i, title in enumerate(recommendations, 1):
        st.write(f"{i}. {title}")
