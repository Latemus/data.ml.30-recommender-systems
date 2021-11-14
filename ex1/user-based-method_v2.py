import pandas as pd
import math
import time


### For performance and accuracy reasons following tresholds can be used:
NUMBER_OF_SIMILAR_USERS_TO_USE = 10         # Number of most similar users to use in the esimation
NUMBER_OF_MOVIE_RECOMMENDATIONS = 20        # How many movies are recommended
NUMBER_OF_NEIGHBORS = 10                    # Number of most similar users to use in the esimation



# In this program a dataset of movie reviews is read from .cvs file and then 
# user-based collaborative filtering is done to predict users rating for a movie
def main():
    TARGET_USER = 15     # User to which we find rating prediction


    # First import the dataset. It contains 4 columns: userId, movieId, rating, and timestamp
    ratings = pd.read_csv('./dataset/ml-latest-small/ratings.csv')
    movies = pd.read_csv('./dataset/ml-latest-small/movies.csv')

    # ratings['usersMeanRating'] = ratings.groupby('userId')['rating'].transform('mean')
    # ratings['movieMeanRating'] = ratings.groupby('movieId')['rating'].transform('mean')
    # ratings['usersRatingCount'] = ratings.groupby('userId')['rating'].transform('count')
    # ratings['movieRatingCount'] = ratings.groupby('movieId')['rating'].transform('count')

    # ratingsWithMovies = ratings.merge(movies, on='movieId')

    print('Number of users: ', len(ratings['userId'].unique()))
    print('Number of movies: ', len(ratings['movieId'].unique()))
    print('Number of ratings: ', len(ratings.index))

    start3 = time.time()
    
    ratingsPerMovie = ratings.pivot_table(index='movieId', columns='userId', values='rating')
    userCorrelationMatrix = ratingsPerMovie.corr('pearson')

    print('ratingsPerMovie:')
    print('ratingsPerMovie index length: ', len(ratingsPerMovie.index))
    print(ratingsPerMovie)


    usersMeanRating = ratings[['userId', 'rating']].copy().rename(columns={'rating': 'usersMeanRating'}).groupby('userId', as_index = False)['usersMeanRating'].mean()
    usersRatingCount = ratings[['userId', 'rating']].copy().rename(columns={'rating': 'usersRatingCount'}).groupby('userId', as_index = False)['usersRatingCount'].count()
    ratings = ratings.merge(usersMeanRating, on='userId').merge(usersRatingCount, on='userId')
    ratings['usersNormilizedRating'] = ratings['rating'] - ratings['usersMeanRating']

    print('ratings')
    print(ratings)

    normilizedRatingsMatrix = ratings.pivot_table(index='userId', columns='movieId', values='usersNormilizedRating')
    print('NormalizedRatingMatrix\n', normilizedRatingsMatrix)

    time3 = time.time() - start3
    print(f'Time: ', time3)

    user1_recommendations = getUsersMovieRecommendations(userCorrelationMatrix, ratingsPerMovie, normilizedRatingsMatrix, ratings, movies, TARGET_USER)
    print(user1_recommendations.head(NUMBER_OF_MOVIE_RECOMMENDATIONS))



# Gets users movie recommendations for films that user has not yet ranked
def getUsersMovieRecommendations(userCorrelationMatrix, ratingsPerMovie, normilizedRatingsMatrix, ratings, movies, targetUser):

    # print(userCorrelationMatrix)

    # print('testing 1076')
    # print(movies[movies['movieId'] == 1076])
    # print('normilizedRatingsMatrix index length: ', len(normilizedRatingsMatrix.columns))
    # print(normilizedRatingsMatrix.loc[0])
    # print(normilizedRatingsMatrix.loc['0'])
    # print(normilizedRatingsMatrix.loc[:, 0])
    # print(normilizedRatingsMatrix.loc[1, '1076'])
    # print(normilizedRatingsMatrix.at[1, '1076'])


    # Get all users rating and after that all movies that user has not yet rated
    usersRatings = ratings.loc[ratings.userId == targetUser]['movieId'].values.tolist()
    movies['prediction'] = 0.0
    unratedMovies = movies[~movies['movieId'].isin(usersRatings)]
    print(unratedMovies)

    print(f'Count of all movies: {len(movies.index)}')
    print(f'Count of movies user {targetUser} has not rated yet: ', len(unratedMovies))

    # print('ratingsPerMovie\n', ratingsPerMovie)
    # print('unratedMovies\n', unratedMovies)

    usersRatingMean = ratings[ratings['userId'] == targetUser]['usersMeanRating'].iloc[0]
    print('usersRatingMean: ', usersRatingMean)

    for movieId in unratedMovies['movieId'].tolist():

        # Select target users row from similarity / correlation matrix
        usersRow = userCorrelationMatrix.loc[targetUser].dropna()

        divident = 0
        divisor = 0
        numberOfNeighboursUsed = 0
        neighbours =  usersRow.sort_values(ascending=False)

        for (userId, correlation) in neighbours.iteritems():
            if (numberOfNeighboursUsed >= NUMBER_OF_NEIGHBORS) | (correlation <= 0):
                break
            
            # print('movie: ', row['movieId'], ', userId: ', userId)
            normilizedRating = normilizedRatingsMatrix.at[userId, movieId]

            if not math.isnan(normilizedRating):
                numberOfNeighboursUsed += 1
                divident += correlation * normilizedRating
                divisor += correlation
            
        if divisor != 0:
            predictedRating = usersRatingMean + divident / divisor
            unratedMovies.at[unratedMovies['movieId'] == movieId, 'prediction'] = predictedRating

    return unratedMovies.sort_values(by='prediction', ascending=False)





def calculatePearsonsCorrelationBetweenUsers_v2(ratings, user_a_id, user_b_id):
    set = ratings[(ratings['userId'] == user_b_id) | (ratings['userId'] == user_a_id)]
    set_pivot = set.pivot_table(index='movieId', columns='userId', values='rating')
    corr = set_pivot.corr('pearson')
    print(corr)
    return corr



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



# # Finds the most similar users to target user that have rated given target movie
# # Returns list of tuples: [(similarity, userId)] or -1. if given user has allready rated given movie
# def findMostSimilarUsers(ratings, targetUser, targetMovie):

#     # First lets get all users id's to a simple list of users that have ranked target movie
#     otherUsersThatHaveRatedTargetMovie = ratings.loc[ratings.movieId == targetMovie].userId.unique().tolist()
#     # print('Number of other users that have review the target film: ', len(otherUsersThatHaveRatedTargetMovie))

#     # If user limit is defined to be over 0, use that. Otherwise this will go through all other users
#     usersLimit = MAX_COUNT_OF_USERS_TO_CHECK if MAX_COUNT_OF_USERS_TO_CHECK > 0 else len(otherUsersThatHaveRatedTargetMovie)

#     # Getting similarity score between targeted and every other user. Tupple of (similarityScore, userId) is returned
#     # Is only done for users, that have rated the target movie (and are not the target user)
#     similarityScores = [(calculatePearsonsCorrelationBetweenUsers(ratings, targetUser, anotherUser), anotherUser) \
#         for anotherUser in otherUsersThatHaveRatedTargetMovie[:usersLimit] if (anotherUser != targetUser)]
    
#     # Sort results and return first n number of users.
#     similarityScores.sort(reverse=True)
#     return similarityScores[:NUMBER_OF_SIMILAR_USERS_TO_USE]





    


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