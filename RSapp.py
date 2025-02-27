import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import base64

# Load dataset
file_path = "net.csv"
df = pd.read_csv(file_path)
df1 = df.dropna()

df1['combined_features'] = df1['genres'] + ',' + df1['type']

# Create TF-IDF Matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df1['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Set Background Image using Base64 Encoding
def set_bg_image(image_file):
    with open(image_file, "rb") as image:
        img_data = image.read()
    b64_img = base64.b64encode(img_data).decode()
    bg_image_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .title-text {{
        color: white;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 20px;
    }}
    .subheader-text {{
        color: white;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
        margin-top: 30px;
        text-decoration: underline;
    }}
    .recommendation {{
        color: white;
        font-size: 24px;
        text-shadow: 1px 1px 2px #000000;
    }}
    .selectbox-label {{
        color: white;
        font-size: 20px;
        font-weight: bold;
        text-shadow: 1px 1px 2px #000000;
    }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Call the function with the image path
set_bg_image('Net.jpg')

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

# Streamlit App Title (in White)
st.markdown('<h1 class="title-text">Movie and TV Series Recommender</h1>', unsafe_allow_html=True)

# Highlighted Selectbox Label (in White)
st.markdown('<label class="selectbox-label">Select a Movie or TV Series for Recommendations:</label>', unsafe_allow_html=True)
selected_title = st.selectbox(
    "",  # Keeping the label empty since it's styled above
    df1['title'].values
)

if st.button("Get Recommendations"):
    recommendations = recommend_movie(selected_title)
    
    # Top 5 Recommendations Subheader (in White)
    st.markdown('<h2 class="subheader-text">Top 5 Recommendations:</h2>', unsafe_allow_html=True)
    
    # Display Recommendations in White
    for i, title in enumerate(recommendations, 1):
        st.markdown(f'<p class="recommendation">{i}. {title}</p>', unsafe_allow_html=True)
