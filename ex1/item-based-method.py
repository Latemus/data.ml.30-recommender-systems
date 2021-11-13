from io import TextIOWrapper
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

pd.set_option("display.precision", 3)
NUMBER_OF_NEIGHBORS = 50                # Number of most similar users to use in the esimation
MIN_NUMBER_OF_NEIGHBOUR_RATINGS = 5     # Minimun number of neighbours that user has rated. This is to increace accuracy
NUM_OF_PREDICTIONS_TO_LIST = 20         # Number of movie predictions to print
TARGET_USER_ID = 414                    # User to which we find rating prediction

def main():
    # First import the dataset of ratings and movies and merge them to together to a dataframe
    ratings = pd.read_csv('./dataset/ml-latest-small/ratings.csv')
    movies = pd.read_csv('./dataset/ml-latest-small/movies.csv')
    ratingsWithMovies = ratings.merge(movies, on='movieId')

    usersRatingCount = ratings.copy().rename(columns={'rating': 'usersRatingCount'}).groupby(['userId'], as_index = False, sort = False)['usersRatingCount'].count()
    print('Top 5 users with most ratings:\n', usersRatingCount.sort_values(by='usersRatingCount', ascending=False).head(5))
    ratingsWithMovies.merge(usersRatingCount, on='userId')

    # Create a pivot table
    pivotedRatingss = ratingsWithMovies.pivot_table(index = 'movieId', columns='userId', values='rating').fillna(0)

    # Initiate NearestNeighbors and fit our pivoted ratings table to it
    nearestNeighbor = NearestNeighbors(metric='cosine', algorithm='brute')
    nearestNeighbor.fit(pivotedRatingss.values)
    n_neighbors = min(len(pivotedRatingss.index), NUMBER_OF_NEIGHBORS +1) # One is added, as the target movie still exists on this set and is removed later
    distances, indices = nearestNeighbor.kneighbors(pivotedRatingss, n_neighbors)
    
    # Add 1 to all values on indices, so they are same as indexes in the pivot table
    indices = indices + 1
    # Modify neighbour data to different format. List of tuples, where [0] is neighbour id and [1] is it's cosine distance
    similarMovies = mergeSimilarities(indices, distances)

    # Create a new dataframe that contains all the movies in the pivotedRatings table. For each movie add it's similarityData to the 'nearestNeighbours' column
    moviesWithNeighbours = pd.DataFrame(pivotedRatingss.index).merge(movies, on='movieId')
    moviesWithNeighbours['nearestNeighbours'] = similarMovies
    moviesWithNeighbours['movieRatingCount'] = ratings.groupby('rating')['rating'].transform('count')
    moviesWithNeighbours['movieMeanRating'] = ratings.groupby('movieId')['rating'].transform('mean')

    predictUsersRatings(ratingsWithMovies, moviesWithNeighbours)



def predictUsersRatings(ratingsWithMovies, moviesWithNeighbours):

    # Get list of movieId's that target user has rated
    usersRatingsList = ratingsWithMovies.loc[ratingsWithMovies['userId'] == TARGET_USER_ID]['movieId'].tolist()
    usersMeanRating = ratingsWithMovies.groupby('userId')['rating'].mean().iloc[0]
    print(f'Users {TARGET_USER_ID} has rated {len(usersRatingsList)} movies. User\'s mean rating is {usersMeanRating}')
    print(f'Now for each unrated movie a prediction is calculated based on ratings that target user has given to it\'s most similar neighbours.')
    print(f'Number of neighbours that each movie has: {NUMBER_OF_NEIGHBORS}. Required number of ratings for neighbours for it to be given prediction: {MIN_NUMBER_OF_NEIGHBOUR_RATINGS}')
    print(f'This will take a while...')

    moviesWithNeighbours['ratingPrediction'] = float(0.0)
    moviesWithNeighbours['numberOfNeighboursWithRatings'] = 0

    # First get all movies that user has not yet rated
    usersRatings = ratingsWithMovies[ratingsWithMovies['userId'] == TARGET_USER_ID]
    unratedMovies = moviesWithNeighbours[~moviesWithNeighbours['movieId'].isin(usersRatingsList)]

    # For each unrated movie we go through it's neighbourhood and calculate predicted rating based on ratings that target user has given to the movies nearest neighbours
    for i, row in unratedMovies.iterrows():
        divident = float(0.0)
        divisor = float(0.0)
        nearestNeighbours = row['nearestNeighbours'] # list of tuples in form of [(movieIndex), (cosineDistance)]
        numberOfNeighboursWithRatings = 0
        for neighbour in nearestNeighbours:
            if neighbour[0] in usersRatingsList:
                numberOfNeighboursWithRatings += 1
                usersRatingForNeighbourMovie = usersRatings[usersRatings['movieId'] == neighbour[0]]['rating'].iloc[0]
                # As the similarities are calculated as cosine distance that is between 0 and 1, we'll calculate similarty as (1 - distance)
                divident += (1- neighbour[1]) * usersRatingForNeighbourMovie
                divisor += (1- neighbour[1])
    
        # User has rated som of the movies in the neighbourhood
        if (divisor != 0) & (numberOfNeighboursWithRatings >= MIN_NUMBER_OF_NEIGHBOUR_RATINGS):
            prediction = divident / divisor
            unratedMovies.at[i, 'ratingPrediction'] = prediction
            unratedMovies.at[i, 'numberOfNeighboursWithRatings'] = numberOfNeighboursWithRatings

    moviePredictions = unratedMovies.drop(columns=['nearestNeighbours'])[unratedMovies['ratingPrediction'] != 0].sort_values(by=['ratingPrediction'], ascending=False)
    moviePredictions['ratingPrediction'] = moviePredictions['ratingPrediction'].round(2)
    moviePredictions['movieMeanRating'] = moviePredictions['movieMeanRating'].round(2)
    usersMeanRating = moviePredictions['ratingPrediction'].mean()
    print(f'Number of movies, that were given a prediction: {len(moviePredictions)}. Mean prediction: {usersMeanRating}')
    print(f'Top {NUM_OF_PREDICTIONS_TO_LIST} movie predictions:')
    print(moviePredictions.head(NUM_OF_PREDICTIONS_TO_LIST))
    return moviePredictions.head(NUM_OF_PREDICTIONS_TO_LIST)
    

# Takes neighbour data (closes neighbour )
def mergeSimilarities(indices, distances):
    listOfTuples = []
    # Go through each neighbour data set as a list. First zip indices and distances to tuple, so that thei can be enumerated at the same time
    # Start from 1, as it is index of first movie in this list.
    for moviesIndex, (indice, distance) in enumerate(zip(indices.tolist(), distances.tolist()), start=1):
        # If movie is as it's own neighbour, remove it
        if moviesIndex in indice:
            index = indice.index(moviesIndex)
            if (index >= 0):
                indice.pop(index)
                distance.pop(index)
        # Add movies neighbours as list of tupples to the main list. Limit them to first n neighbours
        listOfTuples.append(list(zip(indice[:NUMBER_OF_NEIGHBORS], distance[:NUMBER_OF_NEIGHBORS])))
    return listOfTuples



main()
