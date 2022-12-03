from knn_recommender import KNNRecommender
import pandas as pd
import streamlit as st


if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

st.title("Movie Recommendation System")
st.text("CDS Onboarding Project")

movie_id_list = pd.read_csv("datasets/ml-latest-small/ratings.csv")['movieId'].unique()
movie_id_list.sort()
movies_df = pd.read_csv("datasets/ml-latest-small/movies.csv", usecols=['movieId','title'])
movie_name_list = []
movie_name_id = {}
for m_id in movie_id_list:
    m_name = movies_df.loc[movies_df['movieId'] == m_id, 'title'].iloc[0]
    movie_name_list.append(m_name)
    movie_name_id[m_name] = m_id

form = st.form(key="submit-form")
movie_name = form.selectbox(
    'Movie name: ',
    movie_name_list)
rating = form.number_input(label = "Movie rating: ", min_value = 1, max_value = 5)
add = form.form_submit_button("Add to Ratings")

knn_rec = KNNRecommender()

if add:
    movie_id = movie_name_id[movie_name]
    print("Adding " + str(movie_id) + " " + str(rating))
    st.session_state.user_ratings[movie_id] = rating

if st.button(label = "Recommend Movies"):
    st.write("Your Recommended Movies: ")
    knn_recs, _ = knn_rec.recommend(st.session_state.user_ratings)
    st.write(knn_recs)