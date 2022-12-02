from knn_recommender import KNNRecommender
import streamlit as st


if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

st.title("Movie Recommendation System")
st.text("CDS Onboarding Project")
form = st.form(key="submit-form")
movie_id = form.number_input(label = "Movie ID: ", min_value = 1, max_value = 193609)
rating = form.number_input(label = "Movie rating: ", min_value = 1, max_value = 5)
add = form.form_submit_button("Add to Ratings")

knn_rec = KNNRecommender()

if add:
    print("Adding " + str(movie_id) + " " + str(rating))
    st.session_state.user_ratings[movie_id] = rating

if st.button(label = "Recommend Movies"):
    st.write("Your Recommended Movies: ")
    knn_recs = knn_rec.recommend(st.session_state.user_ratings)
    st.write(knn_recs)
