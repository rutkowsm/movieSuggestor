"""
authors: Marcin Rutkowski, Łukasz Reinke
emails: s12497@pjwstk.edu.pl , s15037@pjwstk.edu.pl

Environment configuration:

pip install bs4
pip install numpy
pip install scipy
pip install scikit-learn
"""

import json
import numpy as np
from scipy.spatial import distance


# Load the JSON data into a dictionary
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# Calculate the Euclidean distance between two users
def euclidean_distance(user1, user2, data):
    """
    # Function calculating Euclidean distance.
    #Takes two users and their data (movies and scores) and compares the similarity of those sets within the two users.
    :param user1: one user to compare
    :param user2: other user to compare
    :param data: dictionary with movies and their scores
    :return: float value of Euclidean distance between the users.
    """
    common_movies = set(data[user1]).intersection(set(data[user2]))
    if not common_movies:
        return 0  # Return 0 if no movies in common
    ratings1 = np.array([data[user1][movie] for movie in common_movies])
    ratings2 = np.array([data[user2][movie] for movie in common_movies])
    return np.linalg.norm(ratings1 - ratings2)


# Calculate the cosine distance between two users
def cosine_distance(user1, user2, data):
    """
    # Function calculating Cosine distance.
    #Takes two users and their data (movies and scores) and compares the similarity of those sets within the two users.
    :param user1: one user to compare
    :param user2: other user to compare
    :param data: dictionary with movies and their scores
    :return: float value of Cosine distance between the users.
    """
    common_movies = set(data[user1]).intersection(set(data[user2]))
    if not common_movies:
        return 0  # Return 0 similarity if no movies in common
    ratings1 = np.array([data[user1][movie] for movie in common_movies])
    ratings2 = np.array([data[user2][movie] for movie in common_movies])
    return 1 - distance.cosine(ratings1, ratings2)


# Find similar users based on distance metric
def find_users(target_user, data, distance_func):
    """
    # Function users who are similar to a given target_user based on a specified distance function
    # (which could be either the Euclidean distance or cosine similarity as defined by the distance_func parameter).
    # It then ranks these users by their similarity to the target_user.
    :param target_user: The user that is the focal object of the function.
    :param data: scoring data dictionary with users and their scored movies
    :param distance_func: either Cosine or Euclidean - functions defined above
    :return: sorted list of tuples - users and their scores
    """
    similarities = []
    for user in data:
        if user != target_user:
            dist = distance_func(target_user, user, data)
            similarities.append((user, dist))
    # Sort users by similarity score (lower is more similar for Euclidean, higher for cosine)
    similarities.sort(key=lambda x: x[1], reverse=(distance_func is cosine_distance))
    return similarities


# Recommend movies based on similar users' ratings
# Function to recommend movies based on similar users' ratings
# Recommend movies based on similar users' ratings
def recommend_movies(target_user, data, num_results=5, high_rating_threshold=7, low_rating_threshold=5):
    """
    # Function makes recommendations for movies that a target user might like,
    # based on the viewing habits and preferences of similar and dissimilar users within a dataset.
    # It also identifies movies that are likely to be disliked by the target user.
    # The function has been written to accept dynamic thresholds for what constitutes a "high" and "low" rating,
    # making it adaptable to different datasets.
    :param target_user: Object - The user that is the focal object of the function.
    :param data: dictionary - scoring data dictionary with users and their scored movies
    :param num_results: Integer - number of movies that should be recommended.
    :param high_rating_threshold: Float - threshold of scoring. If equal or higher then a movie is treated as Good
    :param low_rating_threshold: Float - threshold of scoring. If equal or lower then a movie is treated as Bad
    :return: two lists of items - String: Titles; Float: SCore
    """
    # Find users similar to the target user
    similar_users = find_users(target_user, data, cosine_distance)[:5]
    dissimilar_users = find_users(target_user, data, cosine_distance)[-5:]

    # Dictionaries to hold potential recommendations and non-recommendations
    recommended_movies = {}
    not_recommended_movies = {}

    # Dictionaries to hold the count of ratings for calculating the average
    recommended_movies_count = {}
    not_recommended_movies_count = {}

    # Check movies rated by similar users
    for similar_user, _ in similar_users:
        for movie, rating in data[similar_user].items():
            if movie not in data[target_user]:  # Target user hasn't seen/rated it
                if rating >= high_rating_threshold:
                    recommended_movies[movie] = recommended_movies.get(movie, 0) + rating
                    recommended_movies_count[movie] = recommended_movies_count.get(movie, 0) + 1
                else:  # Low-rated movies by similar users
                    not_recommended_movies[movie] = not_recommended_movies.get(movie, 0) + rating
                    not_recommended_movies_count[movie] = not_recommended_movies_count.get(movie, 0) + 1

    # Check movies rated by dissimilar users
    for dissimilar_user, _ in dissimilar_users:
        for movie, rating in data[dissimilar_user].items():
            if movie not in data[target_user] and rating <= low_rating_threshold:
                not_recommended_movies[movie] = not_recommended_movies.get(movie, 0) + rating
                not_recommended_movies_count[movie] = not_recommended_movies_count.get(movie, 0) + 1

    # Calculate the average score for recommended movies
    for movie in recommended_movies:
        recommended_movies[movie] /= recommended_movies_count[movie]

    # Calculate the average score for non-recommended movies
    for movie in not_recommended_movies:
        not_recommended_movies[movie] /= not_recommended_movies_count[movie]

    # Sort the movies based on the average score for recommendations
    sorted_recommendations = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    # Sort the movies based on the average score for non-recommendations
    sorted_non_recommendations = sorted(not_recommended_movies.items(), key=lambda x: x[1])

    # Return the top recommended and top non-recommended movies based on average score
    return sorted_recommendations[:num_results], sorted_non_recommendations[:num_results]


def display_similarities(target_user, data):
    """
    # Function prints out a list of users who are most and least similar to a specified target_user
    # based on their movie ratings within a dataset.
    # It utilizes two different similarity metrics: Euclidean distance and cosine similarity.
    :param target_user: Object - The user that is the focal object of the function.
    :param data: dictionary - scoring data dictionary with users and their scored movies
    :return: none
    """
    print("User\t\t\tSimilarity score euclidean")
    print("-" * 40)
    euclidean_similarities = find_users(target_user, data, euclidean_distance)
    for user, score in euclidean_similarities[:5]:
        print(f"{user}\t\t\t{score:.2f}")

    print("-" * 40)
    print("User\t\t\tUnsimilarity score euclidean")
    print("-" * 40)
    for user, score in euclidean_similarities[-5:]:
        print(f"{user}\t\t\t{score:.2f}")

    print("-" * 40)
    print("User\t\t\tSimilarity score cos")
    print("-" * 40)
    cosine_similarities = find_users(target_user, data, cosine_distance)
    for user, score in cosine_similarities[:5]:
        print(f"{user}\t\t\t{score:.2f}")

    print("-" * 40)
    print("User\t\t\tUnsimilarity score cos")
    print("-" * 40)
    for user, score in cosine_similarities[-5:]:
        print(f"{user}\t\t\t{score:.2f}")


def main(json_data, target_user):
    """
    # function acts as the entry point for the recommendation process in the script.
    # It uses the data and the target user's information to produce and display a list of movie recommendations
    # and non-recommendations based on user preferences.
    # It utilizes all functions described above
    :param json_data: dataset from json file - users and their scorings
    :param target_user: Object - The user that is the focal object of the function.
    :return: none
    """
    # print(f"Finding similar and unsimilar users for {target_user}...\n")
    # display_similarities(target_user, json_data)

    recommended, not_recommended = recommend_movies(target_user, json_data)
    print("-" * 40)
    print("Recommended movies based on similar user preferences:")
    for movie, score in recommended:
        print(f"'{movie}' \tAverage score: {score}")

    print("-" * 40)
    print("\nMovies not recommended based on user preferences:")
    for movie, score in not_recommended:
        print(f"'{movie}' \tAverage score: {score}")


# Main function to run the program

json_data = load_data('dataset.json')  # replace with your JSON filename
target_user = input('Your User Name: ')  # replace with the user you want recommendations for

if __name__ == "__main__":
    main(json_data, target_user)
