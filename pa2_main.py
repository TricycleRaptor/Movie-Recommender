##  Arrin Bevers
##  CPTS375, Fall 2021
##  Due October 17th

from os import name
import pandas as pd #pip install pandas, required for assignment
from sklearn.neighbors import NearestNeighbors # Imported from scikit-learn.org, not stated as required in the assignment but it might be a good idea to use this

def main():
    movies = pd.read_csv("movies.csv")
    links = pd.read_csv("links.csv")
    ratings = pd.read_csv("ratings.csv")
    tags = pd.read_csv("tags.csv") # Pull in data

    df = pd.merge(ratings, movies, on="movieId")

    print("Computing matrices...")

    movie_matrix = df.pivot_table(index="userId", columns="title", values="rating")
    corr_matrix = movie_matrix.corr(method="pearson", min_periods=50) # Centered cosine approach
    total_users = len(df['userId'].value_counts()) # Defines upper bound of the first loop

    for i in range(1, total_users):
        user_ratings = movie_matrix.iloc[i].dropna() # Drop nulls/0's
        print("Evaluating ratings for user " + str(i))
        #print(user_ratings)
        recommend = pd.Series()

        for j in range(0, len(user_ratings)): # Sort neighborhood 5 from similar and add them to recommend, use x
            #print("Adding similar movies to " + user_ratings.index[j])
            similar = corr_matrix[user_ratings.index[j]].dropna() # Drop nulls/0's
            similar = similar.map(lambda x: x * user_ratings[j]) # Finds similarities
            recommend = recommend.append(similar) # Recommendations

        #print("Sorting Recommendations")
        recommend.sort_values(inplace=True, ascending = False)
        #print("User ID: " + str(i))
        #print(recommend.head(5))

        x = pd.DataFrame(recommend)
        recommend_filter = x[~x.index.isin(user_ratings.index)]
        recommend_filter_headresult = recommend_filter[0:5]

        # Debugging nonsense
        #print(recommend_filter_headresult)
        #print(recommend_filter_headresult.index)
        #print(len(recommend_filter_headresult))

        output = str(i)

        for k in list(recommend_filter_headresult.index): #Go into movies, use k to identify moveId
            output = output + " " + str(movies.loc[movies['title'] == k, 'movieId'].iloc[0]) # Acquired movieIds, append to output string

        # Write to file
        results_file = open("results.txt", "a")
        results_file.write(output)
        results_file.write("\n") # More memory efficient
        results_file.close()

    print("Operation finished successfully!\nSee results.txt in assignment directory for output.")

if __name__ == "__main__":
    main()