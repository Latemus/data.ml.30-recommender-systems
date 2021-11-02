import pandas as pd
from sklearn.neighbors import NearestNeighbors



### For performance and accuracy reasons following tresholds can be used:
NUMBER_OF_NEIGHBORS = 30         # Number of most similar users to use in the esimation


# In this program a dataset of movie reviews is read from .cvs file and then 
# user-based collaborative filtering is done to predict users rating for a movie
def cosine_similarity_method():
    TARGET_USER = 1                     # User to which we find rating prediction
    TARGET_MOVIE_NAME = "Toy Story (1995)"    # Movie that we want to predict a rating
    TARGET_MOVIE_ID = 1    # Movie that we want to predict a rating
    
    # First import the dataset. It contains 4 columns: userId, movieId, rating, and timestamp
    ratings = pd.read_csv('./dataset/ml-latest-small/ratings.csv')
    movies = pd.read_csv('./dataset/ml-latest-small/movies.csv')


    ratings = ratings.copy()
    ratingsWithMovies = ratings.merge(movies, on='movieId')
    pivotedRatingss = ratingsWithMovies.pivot_table(index = 'title', columns='userId', values='rating').fillna(0)

    # Initiate NearestNeighbors and fit our pivoted ratings table to it
    nearestNeighbor = NearestNeighbors(metric='cosine', algorithm='brute')
    nearestNeighbor.fit(pivotedRatingss.values)
    distances, indices = nearestNeighbor.kneighbors(pivotedRatingss, n_neighbors = NUMBER_OF_NEIGHBORS + 1) # One is added, as the target movie still exists on this set and is removed later


    targetIndex = pivotedRatingss.index.tolist().index(TARGET_MOVIE_NAME)
    print('targetIndex', targetIndex)

    nearestMovies = indices[targetIndex].tolist()
    distanceToOthers = distances[targetIndex].tolist()
    moviesId = nearestMovies.index(targetIndex)
    # As target movie was not remved from the initial data set, we have to remove it from the results
    # As it would be it's most similar movie with leas amount of distance to itself
    nearestMovies.remove(targetIndex)
    distanceToOthers.pop(moviesId)

    # 
    similarMovies = list(zip(nearestMovies, distanceToOthers))

    # Show the results for the nearest movies and shortest distances
    print(f'The Nearest Movies to {TARGET_MOVIE_NAME} :\n{similarMovies}')

    usersIndex = pivotedRatingss.columns.tolist().index(TARGET_USER)

cosine_similarity_method()



# def correlation_method():
#     TARGET_MOVIE_ID = 1    # Movie that we want to predict a rating

#     # First import the dataset. It contains 4 columns: userId, movieId, rating, and timestamp
#     ratings = pd.read_csv('./dataset/ml-latest-small/ratings.csv')
#     movies = pd.read_csv('./dataset/ml-latest-small/movies.csv')

#     # First for all users calculate mean rating and combine it to ratings dataframe
#     movie_means = ratings.groupby(['movieId'], as_index = False, sort = False)['rating'].mean().rename(columns={'rating': 'movie_rating_mean'})
#     user_means = ratings.groupby(['userId'], as_index = False, sort = False)['rating'].mean().rename(columns={'rating': 'user_rating_mean'})
#     movie_ratings_count = ratings.rename(columns={'rating': 'movie_rating_count'}).groupby(['movieId'], as_index = False, sort = False)['movie_rating_count'].count()
#     ratings = pd.merge(ratings, movie_means, on='movieId')
#     ratings = pd.merge(ratings, user_means, on='userId')
#     ratings = pd.merge(ratings, movie_ratings_count, on='movieId')
#     # For each rating calculate it's nomalized value (rating - rating_mean)
#     ratings['users_normalized_rating'] = ratings['rating'] - ratings['user_rating_mean']

#     ratings = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')

#     print(ratings.head())

#     # Lets create a user x movie pivot table
#     pivotTable = ratings.pivot_table(index = 'userId', columns='movieId', values='rating')

#     print(pivotTable.head)

#     targetMovieRatings = pivotTable[TARGET_MOVIE_ID]

#     movieCorrelations = pivotTable.corrwith(targetMovieRatings)

#     movieCorrelations

#     movieCorrelations = pd.DataFrame(movieCorrelations, columns=['correlation'])
#     movieCorrelations = pd.merge(movieCorrelations, movie_ratings_count, on='movieId')
#     movieCorrelations = movieCorrelations.sort_values(by='correlation', ascending=False)
#     print(movieCorrelations.head())