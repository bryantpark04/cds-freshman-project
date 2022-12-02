from knn_recommender import KNNRecommender

user_ratings = {}

while True:
    try:
        movie_id = int(input("Enter a movie ID (press enter to quit): "))
        rating = int(input("Enter rating for movie " + str(movie_id) + ": "))

        user_ratings[movie_id] = rating
    except:
        break

print("KNN Movie Recommendations:")
knn_rec = KNNRecommender()
print(knn_rec.recommend(user_ratings))