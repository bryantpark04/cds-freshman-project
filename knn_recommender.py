import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from recommender import Recommender


class KNNRecommender(Recommender):
    def __init__(self):
        # Data pre-processing
        self.movies = pd.read_csv("datasets/ml-latest-small/movies.csv") \
            .drop("genres", axis=1)
        ratings = pd.read_csv("datasets/ml-latest-small/ratings.csv") \
            .drop("timestamp", axis=1)

        pivot_table = ratings.pivot(index="movieId", columns="userId", values="rating") \
            .fillna(0)
        self.sparse_matrix = csr_matrix(pivot_table.values)

        # Model training
        self.model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20) \
            .fit(self.sparse_matrix)

        
    def recommend(self, ratings):
        recommendations = set()
        recommended_movies = []

        liked_movies = [movieId for movieId in ratings if ratings[movieId] > 3]
        # TODO: use the rating itself to determine how many movies from that cluster to recommend

        for movie_id in liked_movies:
            _, ids = self.model.kneighbors(self.sparse_matrix[movie_id], n_neighbors=5)
            recommendations.update(*(set(rec_id for rec_id in x if rec_id not in ratings) for x in ids))
            
        for movie_id in recommendations:
            temp = self.movies.loc[self.movies['movieId'] == movie_id, 'title']
            if not temp.empty:
                recommended_movies.append(temp.item())

        return recommended_movies