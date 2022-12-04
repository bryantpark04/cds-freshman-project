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
        self.ratings = pd.read_csv("datasets/ml-latest-small/ratings.csv") \
            .drop("timestamp", axis=1)

        pivot_table = self.ratings.pivot(index="movieId", columns="userId", values="rating") \
            .fillna(0)
        self.sparse_matrix = csr_matrix(pivot_table.values)

        # Model training
        self.model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20) \
            .fit(self.sparse_matrix)
        
        
    def recommend(self, ratings):
        recommendations = set()
        recommended_movies = []

        liked_movies = [movieId for movieId in ratings if ratings[movieId] > 3]

        for movie_id in liked_movies:
            if movie_id > self.movies['movieId'].unique().size:
                continue
            _, ids = self.model.kneighbors(self.sparse_matrix[movie_id], n_neighbors=5)
            recommendations.update(*(set(rec_id for rec_id in x if rec_id not in ratings) for x in ids))
            
        for movie_id in recommendations:
            temp = self.movies.loc[self.movies['movieId'] == movie_id, 'title']
            if not temp.empty:
                recommended_movies.append(temp.item())

        return recommended_movies, recommendations

    
    def getAccuracy(self):
        # for each user, get random sampling of movies and ratings
            # get recommendations based on random sampling
            # find user's average rating for the returned movies
        for user in self.ratings['userId'].unique():
            user_ratings = self.ratings.loc[self.ratings['userId'] == user].drop('userId', axis = 1)
            user_ratings = user_ratings.sample(n=5)

            user_dict = {}
            for _, row in user_ratings.iterrows():
                if(row['movieId'] >= 9724):
                    continue
                user_dict[row['movieId']] = row['rating']
            
            _, recommendations = self.recommend(user_dict)
            
            total = 0
            for rec in recommendations:
                temp = (user_ratings.loc[user_ratings['movieId'] == rec, 'rating'])
                if temp.size != 0:
                    total += temp[0]
            
            avg_rating = 0 if len(recommendations) == 0 else total/len(recommendations)

            if(avg_rating != 0):
                print("Average rating for user " + str(user) + ": " + str(avg_rating))


# knn_rec1 = KNNRecommender()
# knn_rec1.getAccuracy()
# print(knn_rec1.sparse_matrix.shape)
# print(knn_rec1.ratings['movieId'].unique().size)