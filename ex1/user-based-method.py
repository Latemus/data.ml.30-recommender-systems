import pandas as pd
from math import sqrt


### For performance and accuracy reasons following tresholds can be used:
NUMBER_OF_SIMILAR_USERS_TO_USE = 10         # Number of most similar users to use in the esimation
MIN_COMMON_RATINGS_BETWEEN_USERS = 0        # (FOR ACCURACITY) to only take into account other users, that have at least this much common ratings with target user
MAX_COUNT_OF_USERS_TO_CHECK = 0             # (FOR TESTING PERFORMANCE) to only go through first n amount of users in the dataset
MAX_COUNT_OF_MOVIES_TO_CHECK = 50           # (FOR TESTING PERFORMANCE) to only go through first n amount of movies in the dataset
NUMBER_OF_MOVIE_RECOMMENDATIONS = 20        # How many movies are recommended



# In this program a dataset of movie reviews is read from .cvs file and then 
# user-based collaborative filtering is done to predict users rating for a movie
def main():
    TARGET_USER = 15     # User to which we find rating prediction
    TARGET_MOVIE = 1    # Movie that we want to predict a rating
    PEARSONS_CORRELATION_OTHER_USER = 3 # Other user to which pearsons correlation is calculated below


    # First import the dataset. It contains 4 columns: userId, movieId, rating, and timestamp
    ratings = pd.read_csv('./dataset/ml-latest-small/ratings.csv')
    movies = pd.read_csv('./dataset/ml-latest-small/movies.csv')
    
    # toyStoryId = movies[movies.title == 'Toy Story (1995)'].movieId.iloc[0]
    # print(toyStoryId)

    # Print five first lines of this set to check that we have right dataset and then count of rows it contains
    print('First five rows of this dataset:')
    print(ratings.head(5))
    print(f'Number of rows in dataset:  {len(ratings.index)}\n')

    # For testing: We can check pearsons correlation for pair of users. NOTE: MIN_COMMON_RATINGS_BETWEEN_USERS affects this
    correlation = calculatePearsonsCorrelationBetweenUsers(ratings, TARGET_USER, PEARSONS_CORRELATION_OTHER_USER)
    print(f'For testing: Pearson\'s correlation between user {TARGET_USER} and {PEARSONS_CORRELATION_OTHER_USER}: {correlation}\n')

    print(f'Find target users {TARGET_USER} most similar users for movie {TARGET_MOVIE}')
    printSimilarUsersAsList(findMostSimilarUsers(ratings, TARGET_USER, TARGET_MOVIE))

    # This function call starts this process of finding users rating prediction for a film
    predictedRating = predictUsersRating(ratings, TARGET_USER, TARGET_MOVIE)
    print(f'\nUsers {TARGET_USER} predicted rating for movie {TARGET_MOVIE}: {predictedRating}\n')

    # Get users recommendation for all unrated films (limited by MAX_COUNT_OF_MOVIES_TO_CHECK)
    getUsersMovieRecommendations(movies, ratings, TARGET_USER)



# Calculates users prediction for a given film
# Returns a numeric prediction or -1 in case of exception
def predictUsersRating(ratings, targetUser, targetMovie):

    # First we check that user has not yet rated target movvie. If so, return -1
    targetUsersRating = getUsersRatingForMovie(ratings, targetUser, targetMovie)
    if targetUsersRating >= 0:
        print(f'Target user has already rated target movie: {targetMovie} with rating: {targetUsersRating}')
        return targetUsersRating

    # First for each other user, that have rated the target movie, a similarity score is calculated
    similarUsers = findMostSimilarUsers(ratings, targetUser, targetMovie)

    # Get target users mean rating score
    targetUsersRatingMean = getUsersRatingMean(ratings, targetUser)

    # Calculate sum of similar users weighted and normilized ratings. Similarity score is used as weight.
    # similarUsers: [(similarityScore, userId)]
    sumOfSimilarUsersWeightedNormilizedRatings = 0
    sumOfSimilarityScores = 0
    for similarUser in similarUsers:
        usersNormalizedRating = getUsersRatingForMovie(ratings, similarUser[1], targetMovie) - getUsersRatingMean(ratings, similarUser[1])
        sumOfSimilarUsersWeightedNormilizedRatings += similarUser[0] * usersNormalizedRating
        sumOfSimilarityScores += similarUser[0]

    if sumOfSimilarityScores == 0:
        return -1
    
    # Calculate similarity rating. This formula is taken from courses second lecture slides
    return targetUsersRatingMean + (sumOfSimilarUsersWeightedNormilizedRatings / sumOfSimilarityScores)



# Finds the most similar users to target user that have rated given target movie
# Returns list of tuples: [(similarity, userId)] or -1. if given user has allready rated given movie
def findMostSimilarUsers(ratings, targetUser, targetMovie):

    # First lets get all users id's to a simple list of users that have ranked target movie
    otherUsersThatHaveRatedTargetMovie = ratings.loc[ratings.movieId == targetMovie].userId.unique().tolist()
    # print('Number of other users that have review the target film: ', len(otherUsersThatHaveRatedTargetMovie))

    # If user limit is defined to be over 0, use that. Otherwise this will go through all other users
    usersLimit = MAX_COUNT_OF_USERS_TO_CHECK if MAX_COUNT_OF_USERS_TO_CHECK > 0 else len(otherUsersThatHaveRatedTargetMovie)

    # Getting similarity score between targeted and every other user. Tupple of (similarityScore, userId) is returned
    # Is only done for users, that have rated the target movie (and are not the target user)
    similarityScores = [(calculatePearsonsCorrelationBetweenUsers(ratings, targetUser, anotherUser), anotherUser) \
        for anotherUser in otherUsersThatHaveRatedTargetMovie[:usersLimit] if (anotherUser != targetUser)]
    
    # Sort results and return first n number of users.
    similarityScores.sort(reverse=True)
    return similarityScores[:NUMBER_OF_SIMILAR_USERS_TO_USE]



# Calculate pearsons correlation between given users on a dataset of ratings
# Excpects dataset (DataFrame object) to have columns 'userId', 'movieId', 'rating'
# Based on formula in Falk, K. (2019). Practical Recommender Systems. Manning Publications Co. LLC.
def calculatePearsonsCorrelationBetweenUsers(ratings, user_a_id, user_b_id):
    # Get a dataframe of all movies user a has rated and a simple list of all the id's
    a_ratings = ratings.loc[ratings.userId == user_a_id]
    b_ratings = ratings.loc[ratings.userId == user_b_id]

    # Create two dataframes, which both contain only movies that both users have rarated
    setA = a_ratings[a_ratings.movieId.isin(b_ratings.movieId.values.tolist())].sort_values(by=['movieId'])
    setB = b_ratings[b_ratings.movieId.isin(a_ratings.movieId.values.tolist())].sort_values(by=['movieId'])

    # If users have no common movies or if count of common ratings is less than threshold, return -1
    if (len(setA.index) == 0 | len(setA.index) < MIN_COMMON_RATINGS_BETWEEN_USERS):
        return -1

    # Combined dataframe for both users ratings is merged here. Contains columns 'movieId', 'rating_a', 'rating_b'.
    combinedSet = pd.merge(
        setA.rename(columns={'rating': 'rating_a'})[['movieId', 'rating_a']], 
        setB.rename(columns={'rating': 'rating_b'})[['movieId', 'rating_b']],
        on = 'movieId')[['rating_a', 'rating_b']]

    user_a_mean_rating = getUsersRatingMean(ratings, user_a_id)
    user_b_mean_rating = getUsersRatingMean(ratings, user_b_id)

    # Calculate sum of products of normilized ratings
    sumOfNormilizedRatingsProducts = 0
    for i, row in combinedSet.iterrows():
        a_norm = row.rating_a - user_a_mean_rating
        b_norm = row.rating_b - user_b_mean_rating
        sumOfNormilizedRatingsProducts += (a_norm * b_norm)

    # Divider is product of each users calculated sum of normalized ratings from the set of common ratings
    divider = calculateSumOfNormalizedRatingsFromFilteredSet(setA, user_a_mean_rating) * \
        calculateSumOfNormalizedRatingsFromFilteredSet(setB, user_b_mean_rating)
    
    if divider == 0:
        return 0
    
    return sumOfNormilizedRatingsProducts / divider


# Gets users movie recommendations for films that user has not yet ranked
def getUsersMovieRecommendations(movies, ratings, targetUser):
    # Get all users rating and after that all movies that user has not yet rated
    usersRatings = ratings.loc[ratings.userId == targetUser].movieId.values.tolist()
    unratedMovies = movies[~movies.movieId.isin(usersRatings)]

    print(f'Count of all movies: {len(movies.index)}')
    print(f'Count of movies user {targetUser} has not rated yet: ', len(unratedMovies))
    print('Calculating recommendations for movies. This might take a while depending on the hardware...')

    # If user limit is defined to be over 0, use that. Otherwise this will go through all movies
    movieLimit = MAX_COUNT_OF_MOVIES_TO_CHECK if MAX_COUNT_OF_MOVIES_TO_CHECK > 0 else len(unratedMovies)

    # Counts ratings for all movies.
    # Return tuple (predictedRating, movieId, movieTitle)
    movieRecommendations = [(predictUsersRating(ratings, targetUser, movie.movieId), movie.movieId, movie.title) \
        for i, movie in unratedMovies.head(movieLimit).iterrows()]
    
    # Sort returned recommendations and return first 
    movieRecommendations.sort(reverse=True)
    printMovieRecommendations(movieRecommendations[:NUMBER_OF_MOVIE_RECOMMENDATIONS])
    


# Returns rating given by userId to movie movieId.
# Returns value of the rating or -1 if given user has not rated given movie
def getUsersRatingForMovie(all_ratings, userId, movieId):
    rating = all_ratings[(all_ratings.userId == userId) & (all_ratings.movieId == movieId)]
    if (len(rating.index) == 0):
        return -1
    return rating.rating.iloc[0]



# Returns mean score for all ratings given by userId
def getUsersRatingMean(all_ratings, userId):
    usersRatings = all_ratings.loc[all_ratings.userId == userId]
    return usersRatings.rating.mean()



# Calculates sum of users nomalized ratings in a dataset
def calculateSumOfNormalizedRatingsFromFilteredSet(usersRatings, usersMeanRating):
    sumOfNormalizedRatings = 0
    for i, row in usersRatings.iterrows():
        sumOfNormalizedRatings += ((row.rating - usersMeanRating) ** 2) 
    return sqrt(sumOfNormalizedRatings)



# Prints similarUsers as a list
def printSimilarUsersAsList(similarUsers):
    print('Most similar users that have rated target film:')
    for similarUser in similarUsers:
        print(f'UserId: {similarUser[1]},\t similarity score: {similarUser[0]}')

# Prints movieRecommendations as a list
def printMovieRecommendations(movieRecommendations):
    print('\nRecommended movies:\n id, predicted rating, title')
    for movie in movieRecommendations:
        print(f'MovieId: {movie[1]}\tpredicted rating: {movie[0]}\ttitle: {movie[2]}')


main()