# Assignment 1: Collaborative Filtering Recommendations


### User based
Function for user based collaborative filtering using pearson's correlation is in file 'user-based-method.py'. This method uses python library Pandas. It is a solution that works but as it is written quite manually it is very unoptimized and way too slow to handle big datasets. That's why there is a variable 'NUMBER_OF_MOVIE_RECOMMENDATIONS' which limits the amount of movies that are included in the recommendations calulations.

In the main function:
- data is fetched and first rows are shown
- For demo, Pearson's correlation is calculated for two different users and result is shown
- For demo users predicted rating is calculated for a movie and then shown
- Lastly, users top recommendations are calculated and shown.

### Item based
Here idea was to make whole process more optimized and faster. In this solution, only the most similar films are found for the target film and their distance is listed. No recommendations are given based on this.