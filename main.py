movie_ids = []
ratings = []

while True:
    try:
        movie_id = int(input("Enter a movie ID (press enter to quit): "))
        rating = int(input("Enter rating for movie " + str(movie_id) + ": "))

        movie_ids.append(movie_id)
        ratings.append(rating)
    except:
        break

print("Movie IDs: ", movie_ids)
print("Ratings: ", ratings)