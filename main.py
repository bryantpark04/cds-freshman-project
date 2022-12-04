import pandas as pd
import streamlit as st

from knn_recommender import KNNRecommender
from nn_recommender import NN_Recommender


# define state variables
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

if 'knn_rec' not in st.session_state:
	st.session_state.knn_rec = KNNRecommender()

if 'nn_rec' not in st.session_state:
    st.session_state.nn_rec = NN_Recommender()

if 'movie_id_list' not in st.session_state:
    st.session_state.movie_id_list = pd.read_csv("datasets/ml-latest-small/ratings.csv")['movieId'].unique()
    st.session_state.movie_id_list.sort()

if 'movies_df' not in st.session_state:
    st.session_state.movies_df = pd.read_csv("datasets/ml-latest-small/movies.csv", usecols=['movieId','title'])

if 'movie_name_list' not in st.session_state:
    st.session_state.movie_name_list = []
    for m_id in st.session_state.movie_id_list:
        m_name = st.session_state.movies_df.loc[st.session_state.movies_df['movieId'] == m_id, 'title'].iloc[0]
        st.session_state.movie_name_list.append(m_name)

if 'movie_name_id' not in st.session_state:
    st.session_state.movie_name_id = {}
    for m_id in st.session_state.movie_id_list:
        m_name = st.session_state.movies_df.loc[st.session_state.movies_df['movieId'] == m_id, 'title'].iloc[0]
        st.session_state.movie_name_id[m_name] = m_id


# basic titles
st.title("Movie Recommendation System")
st.text("CDS Onboarding Project")

# form to add ratings
form = st.form(key="submit-form")
movie_name = form.selectbox(
    'Movie name: ',
    st.session_state.movie_name_list)
rating = form.number_input(label = "Movie rating: ", min_value = 1, max_value = 5)
add = form.form_submit_button("Add to Ratings")

# add user rating
if add:
    movie_id = st.session_state.movie_name_id[movie_name]
    print("Adding " + str(movie_id) + " " + str(rating))
    st.session_state.user_ratings[movie_id] = rating

# generate recommendations
if st.button(label = "Recommend Movies"):
    st.write("Your Recommended Movies (KNN): ")
    knn_recs, _ = st.session_state.knn_rec.recommend(st.session_state.user_ratings)
    st.write(knn_recs)

    st.write("Your Recommended Movies (NN): ")
    nn_recs = st.session_state.nn_rec.recommend(st.session_state.user_ratings)
    st.write(nn_recs)