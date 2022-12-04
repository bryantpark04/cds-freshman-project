import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn

from recommender import Recommender


class NN_Recommender(Recommender):
    def __init__(self):
        # defining constants
        self.n_input, self.n_hidden, self.n_out, self.batch_size, self.learning_rate = 19, 20, 1, 3, 0.05

        # process data
        self.movies = pd.read_csv("datasets/ml-latest-small/movies.csv")
        self.genres = ['Action', 'Adventure', 'Animation', "Children", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
        self.movies_to_genres = pd.Series(self.movies.genres.values,index=self.movies.movieId).to_dict()

        # model setup
        self.model = nn.Sequential(nn.Linear(self.n_input, self.n_hidden),
                      nn.ReLU(),
                      nn.Linear(self.n_hidden, self.n_out),
                      nn.Sigmoid())
        
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)


    # generate recommended movies based on user ratings
    def recommend(self, ratings):
        if len(ratings) < 3:
            return []
        data_x, data_y = self.process_data(ratings)
        self.train_model(data_x, data_y)

        # for some random sample of 100 movies, add to recommended if not already rated and model predicts value above a threshold
        recommendations = []
        for cur_movie in random.sample(self.movies['movieId'].values.tolist(), 100):
            if cur_movie in ratings.keys():
                continue
            cur_genres = torch.tensor([self.movie_to_genres_arr(cur_movie)])
            cur_genres = cur_genres.to(torch.float32)
            if self.model(cur_genres) > 0.7:
                recommendations.append(cur_movie)

        # convert recommendations to movie names
        recommended_movies = []
        for movie_id in recommendations:
            temp = self.movies.loc[self.movies['movieId'] == movie_id, 'title']
            if not temp.empty:
                recommended_movies.append(temp.item())
        
        return random.sample(recommended_movies, 10) if len(recommended_movies) > 10 else recommended_movies


    # convert user ratings into tensors
    def process_data(self, ratings):
        x = []
        y = []
        for cur_movie, cur_rating in ratings.items():
            # create array based on genres
            cur_genres = self.movie_to_genres_arr(cur_movie)

            # add array to x
            x.append(cur_genres)

            # add value to y
            y.append([1 if cur_rating > 3 else 0])

        # convert lists to tensors
        data_x = torch.tensor(x)
        data_x = data_x.to(torch.float32)
        data_y = torch.tensor(y)
        data_y = data_y.to(torch.float32)

        return data_x, data_y


    # train the neural network
    def train_model(self, data_x, data_y):
        for epoch in range(100):
            pred_y = self.model(data_x)
            loss = self.loss_function(pred_y, data_y)

            self.model.zero_grad()
            loss.backward()

            self.optimizer.step()


    # convert movie into array indicating whether the movie is classified as each genre
    def movie_to_genres_arr(self, movie_id):
        cur_genres = [0] * 19
        for i in range(len(self.genres)):
            if self.genres[i] in self.movies_to_genres[movie_id]:
                cur_genres[i] = 1
        return cur_genres


# nn_rec = NN_Recommender()
# print(nn_rec.recommend({1:4, 3:5, 4:2}))