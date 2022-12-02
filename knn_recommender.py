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

        
    def recommend(self, ratings: dict[int, int]) -> list[str]:
        recommendations = set()

        liked_movies = [movieId for movieId in ratings if ratings[movieId] > 3]
        # TODO: use the rating itself to determine how many movies from that cluster to recommend

        for movie_id in liked_movies:
            _, ids = self.model.kneighbors(self.sparse_matrix[movie_id], n_neighbors=5)
            recommendations.update(*(set(rec_id for rec_id in x if rec_id not in ratings) for x in ids))
            # TODO: return strings and not movieIds
            # recommendations.update({self.movies[self.movies["movieId"] == rec_id]["title"].iloc[0] for rec_id in ids[0] if rec_id not in ratings})

        return recommendations


print(KNNRecommender().recommend({3: 5}))