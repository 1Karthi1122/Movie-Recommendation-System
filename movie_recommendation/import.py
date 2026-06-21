import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import json
from bs4 import BeautifulSoup
import requests, io
import numpy as np
from urllib.request import urlopen
import PIL.Image
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- KNN CLASS --------------------
class KNearestNeighbours:
    def _init_(self, data, target, query_point, k=5):
        self.data = data
        self.target = target
        self.query_point = query_point
        self.k = k

    def fit(self):
        similarities = cosine_similarity([self.query_point], self.data)[0]
        self.indices = np.argsort(similarities)[::-1][:self.k]

# -------------------- LOAD DATA --------------------
with open(r'C:\\Users\\HP\\Desktop\\pavi 1st project\\modified_movie_data.json', 'r', encoding='utf-8') as f:
    full_data = json.load(f)
with open(r'C:\\Users\\HP\\Desktop\\pavi 1st project\\modified_movie_titles.json', 'r', encoding='utf-8') as f:
    full_titles = json.load(f)

# Organize with industry tagging
data = []
movie_titles = []
industries = set()

for i in range(len(full_titles)):
    title, industry, link = full_titles[i]
    industries.add(industry)
    data.append(full_data[i])
    movie_titles.append([title, industry, link])
hdr = {'User-Agent': 'Mozilla/5.0'}

# Get unique industries
industries = sorted(list(set(title[i] for title in movie_titles if len(title)>i)))
selected_industries = st.multiselect('Select Industries to Include:', industries, default=industries)


# -------------------- FETCH MOVIE INFO --------------------
def get_movie_info(imdb_link):
    url_data = requests.get(imdb_link).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    rating_element = s_data.find("span", itemprop="ratingValue")
    rating = rating_element.text.strip() if rating_element else "Rating not found"
    return rating

# -------------------- FETCH MOVIE POSTER --------------------
def movie_poster_fetcher(imdb_link):
    url_data = requests.get(imdb_link, headers=hdr).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_dp = s_data.find("meta", property="og:image")
    if imdb_dp is not None:
        movie_poster_link = imdb_dp.attrs['content']
        u = urlopen(movie_poster_link)
        raw_data = u.read()
        image = PIL.Image.open(io.BytesIO(raw_data))
        image = image.resize((158, 301), )
        st.image(image, use_column_width=False)
    else:
        st.error("Movie poster not found.")

# -------------------- KNN RECOMMENDER FUNCTION --------------------
def KNN_Movie_Recommender(test_point, k):
    target = [0 for item in movie_titles]
    model = KNearestNeighbours(data, target, test_point, k=k)
    model.fit()
    table = []
    for i in model.indices:
        table.append([movie_titles[i][0], movie_titles[i][2], data[i][-1]])
    return table

# -------------------- STREAMLIT SETUP --------------------
st.set_page_config(page_title="Movie Recommender System")
st.title("Welcome to the Movie Recommender System")

def run():
    img1 = Image.open(r'C:\\Users\\HP\\Desktop\\pavi 1st project\\cinema.jpg')
    img1 = img1.resize((250, 250), )
    st.image(img1, use_column_width=False)
    st.title("Movie Recommender System")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based on "IMDB 5000 Movie Dataset"</h4>''', unsafe_allow_html=True)

    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
              'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    filtered_titles = [title for title in movie_titles if title[1] in selected_industries]
    movies = [title[0] for title in filtered_titles]

    category = ['--Select--', 'Movie based', 'Genre based']

    cat_op = st.selectbox('Select Recommendation Type', category)

    if cat_op == category[0]:
        st.warning('Please select Recommendation Type!!')

    elif cat_op == category[1]:  # Movie based
        select_movie = st.selectbox('Select movie: (Recommendation will be based on this selection)', ['--Select--'] + movies)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Fetching a Movie Poster will take some time.</h4>''', unsafe_allow_html=True)

        if select_movie != '--Select--':
            no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
            genres_vector = data[movies.index(select_movie)]
            test_point = genres_vector
            table = KNN_Movie_Recommender(test_point, no_of_reco + 1)
            table.pop(0)
            c = 0
            st.success('Some of the movies from our Recommendation, have a look below')
            for movie, link, ratings in table:
                c += 1
                st.markdown(f"({c}) [ {movie}]({link})")
                if dec == 'Yes':
                    movie_poster_fetcher(link)
                st.markdown('IMDB Rating: ' + str(ratings) + ' ⭐')
        else:
            st.warning('Please select Movie!!')

    elif cat_op == category[2]:  # Genre based
        sel_gen = st.multiselect('Select Genres:', genres)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Fetching a Movie Poster will take some time.</h4>''', unsafe_allow_html=True)

        if sel_gen:
            imdb_score = st.slider('Choose IMDb score:', 1, 10, 8)
            no_of_reco = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
            test_point = [1 if genre in sel_gen else 0 for genre in genres]
            test_point.append(imdb_score)
            table = KNN_Movie_Recommender(test_point, no_of_reco)
            c = 0
            st.success('Some of the movies from our Recommendation, have a look below')
            for movie, link, ratings in table:
                c += 1
                st.markdown(f"({c}) [ {movie}]({link})")
                if dec == 'Yes':
                    movie_poster_fetcher(link)
                st.markdown('IMDB Rating: ' + str(ratings) + ' ⭐')

run()