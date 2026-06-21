import streamlit as st
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- KNN LOGIC --------------------
class KNearestNeighbours:
    def __init__(self, data, query_point, k=5):
        self.data = np.array(data)
        self.query_point = np.array(query_point).reshape(1, -1)
        self.k = k

    def fit(self):
        # Cosine similarity is better for high-dimensional genre data
        similarities = cosine_similarity(self.query_point, self.data)[0]
        # Get indices of top K highest similarity scores
        self.indices = np.argsort(similarities)[::-1][:self.k]
        return self.indices

# -------------------- HELPER FUNCTIONS --------------------
def KNN_Movie_Recommender(test_point, k, movie_data, movie_titles):
    engine = KNearestNeighbours(movie_data, test_point, k)
    indices = engine.fit()
    
    recommendations = []
    for i in indices:
        # movie_titles[i] = [Title, ID, IMDb_Link]
        # movie_data[i][-1] is the IMDb Score
        recommendations.append((movie_titles[i][0], movie_titles[i][2], movie_data[i][-1]))
    return recommendations

def movie_poster_fetcher(imdb_link):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    try:
        url_data = requests.get(imdb_link, headers=hdr).text
        s_data = BeautifulSoup(url_data, 'html.parser')
        # Fetching the poster from Open Graph meta tags
        meta_tag = s_data.find("meta", property="og:image")
        if meta_tag:
            return meta_tag.get('content')
    except Exception:
        return None

# -------------------- APP UI --------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Sidebar/Header Image
try:
    img = Image.open('cinema.jpg')
    st.image(img, use_container_width=True)
except:
    st.title("🎬 Movie Recommender")

# Load Data (Assuming files are in the same directory)
@st.cache_data
def load_data():
    with open('modified_movie_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('modified_movie_titles.json', 'r', encoding='utf-8') as f:
        titles = json.load(f)
    return data, titles

movie_data, movie_titles = load_data()

genres = ['Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'Thriller', 'Documentary', 'Romance', 'Animation', 'Comedy', 'Family', 'Musical', 'Mystery', 'Western', 'Drama', 'History', 'Sport', 'Crime', 'Horror', 'War', 'Biography', 'Music', 'Game-Show', 'Reality-TV', 'News', 'Short', 'Film-Noir']

cat_op = st.selectbox('Recommendation Mode:', ['--Select--', 'Movie Based', 'Genre Based'])

# --- GENRE BASED ---
if cat_op == 'Genre Based':
    sel_gen = st.multiselect('Select Genres:', genres)
    imdb_score = st.slider('Minimum IMDb score:', 1.0, 10.0, 7.0)
    no_of_reco = st.number_input('Number of movies:', 5, 20, 5)
    
    if st.button('Recommend'):
        # Construct test point: 26 genre binary flags + 1 score
        test_point = [1 if g in sel_gen else 0 for g in genres]
        test_point.append(imdb_score)
        
        results = KNN_Movie_Recommender(test_point, no_of_reco, movie_data, movie_titles)
        
        st.success('Top Picks for You:')
        for movie, link, rating in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                poster = movie_poster_fetcher(link)
                if poster: st.image(poster)
            with col2:
                st.subheader(movie)
                st.write(f"⭐ IMDb: {rating}")
                st.write(f"🔗 [Open IMDb]({link})")
            st.divider()

# --- MOVIE BASED ---
elif cat_op == 'Movie Based':
    all_titles = [m[0] for m in movie_titles]
    select_movie = st.selectbox('Search for a movie:', all_titles)
    no_of_reco = st.number_input('Number of recommendations:', 5, 20, 5)

    if st.button('Find Similar'):
        idx = all_titles.index(select_movie)
        test_point = movie_data[idx]
        
        # We fetch k+1 because the top result will be the movie itself
        results = KNN_Movie_Recommender(test_point, no_of_reco + 1, movie_data, movie_titles)
        
        st.success(f"Since you liked {select_movie}:")
        for movie, link, rating in results:
            if movie == select_movie: continue
            st.subheader(movie)
            st.write(f"⭐ IMDb: {rating} | [Link]({link})")
            st.divider()