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
    common_movies = set(data[user1]).intersection(set(data[user2]))
    if not common_movies:
        return float('inf')  # Return infinity if no movies in common
    ratings1 = np.array([data[user1][movie] for movie in common_movies])
    ratings2 = np.array([data[user2][movie] for movie in common_movies])
    return np.linalg.norm(ratings1 - ratings2)


# Calculate the cosine distance between two users
def cosine_distance(user1, user2, data):
    common_movies = set(data[user1]).intersection(set(data[user2]))
    if not common_movies:
        return 0  # Return 0 similarity if no movies in common
    ratings1 = np.array([data[user1][movie] for movie in common_movies])
    ratings2 = np.array([data[user2][movie] for movie in common_movies])
    return 1 - distance.cosine(ratings1, ratings2)


# Find similar users based on distance metric
def find_users(target_user, data, distance_func):
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
def recommend_movies(target_user, data, num_recommendations=5):
    # Find users similar to the target user
    similar_users = find_users(target_user, data, cosine_distance)[:5]
    dissimilar_users = find_users(target_user, data, cosine_distance)[-5:]

    # Dictionaries to hold potential recommendations and non-recommendations
    recommended_movies = {}
    not_recommended_movies = {}

    # Check movies rated by similar users
    for similar_user, _ in similar_users:
        for movie, rating in data[similar_user].items():
            if movie not in data[target_user]:  # Target user hasn't seen/rated it
                if rating >= 8:  # Assuming 8 is the threshold for a high rating
                    recommended_movies[movie] = recommended_movies.get(movie, 0) + rating
                else:  # Low-rated movies by similar users
                    not_recommended_movies[movie] = not_recommended_movies.get(movie, 0) + rating

    # Check movies rated by dissimilar users
    for dissimilar_user, _ in dissimilar_users:
        for movie, rating in data[dissimilar_user].items():
            if movie not in data[target_user]:  # Target user hasn't seen/rated it
                if rating <= 5:  # Assuming 5 is the threshold for a low rating
                    not_recommended_movies[movie] = not_recommended_movies.get(movie, 0) + rating

    # Sort the movies based on the aggregated score for recommendations
    sorted_recommendations = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    # Sort the movies based on the aggregated score for non-recommendations
    sorted_non_recommendations = sorted(not_recommended_movies.items(), key=lambda x: x[1], reverse=False)

    # Return the top recommended and top 5 non-recommended movies
    return sorted_recommendations[:num_recommendations], sorted_non_recommendations[:5]


def display_similarities(target_user, data):
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
    print(f"Finding similar and dissimilar users for {target_user}...\n")
    display_similarities(target_user, json_data)

    recommended, not_recommended = recommend_movies(target_user, json_data)
    print("-" * 40)
    print("Recommended movies based on similar user preferences:")
    for movie, score in recommended:
        print(f"{movie}: {score}")

    print("-" * 40)
    print("\nMovies not recommended based on user preferences:")
    for movie, score in not_recommended:
        print(f"{movie}: {score}")


# Main function to run the program

json_data = load_data('dataset.json')  # replace with your JSON filename
target_user = input('Your User Name: ')  # replace with the user you want recommendations for

if __name__ == "__main__":
    main(json_data, target_user)
