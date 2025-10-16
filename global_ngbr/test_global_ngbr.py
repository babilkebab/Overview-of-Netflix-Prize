import pickle
import numpy as np
from sklearn.metrics import root_mean_squared_error
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_probe


def test_global_ngbr():
    global csr_no_probe, csr_utility_matrix, weights, user_biases_no_probe, movie_biases_no_probe, global_avg, implicit_weights, trained_movie_biases, trained_user_biases

    elapsed = 0
    probe_pred_ratings = np.zeros(1408395)
    probe_true_ratings = np.zeros(1408395)

    for user in probe_movies_rated_by_user.keys():
        user_row = csr_no_probe[user]
        user_row_w_probe = csr_utility_matrix[user]
        user_rated_movies = user_row.indices
        true_ratings = user_row.data
        dense_true_ratings = np.squeeze(np.asarray(user_row_w_probe.todense()))
        user_bias_on_all_movies = global_avg + movie_biases_no_probe[user_rated_movies] + user_biases_no_probe[user]
        centered_ratings = true_ratings - user_bias_on_all_movies
        inv_sqrt = 1.0 / np.sqrt(len(user_rated_movies))
        for movie in probe_movies_rated_by_user[user]:
            bias_user_on_movie = global_avg + trained_user_biases[user] + trained_movie_biases[movie]

            self_mask = user_rated_movies == movie

            current_weights = weights[movie, user_rated_movies]
            current_implicit = implicit_weights[movie, user_rated_movies]

            current_weights[self_mask] = 0
            current_implicit[self_mask] = 0


            pred_rating = bias_user_on_movie + inv_sqrt * (current_weights.dot(centered_ratings) + current_implicit.sum())
            probe_pred_ratings[elapsed] = pred_rating
            probe_true_ratings[elapsed] = dense_true_ratings[movie]


            elapsed+=1
            if elapsed % 100000 == 0:
                print(elapsed)


    rmse = root_mean_squared_error(probe_true_ratings, np.clip(probe_pred_ratings, 1.0, 5.0))
    print(f"RMSE with weighted and biased CF: {rmse}")











if __name__ == "__main__":


    print("Loading structures...")
    with open("../structures/csr_no_probe.pkl", "rb") as csr_file:
        csr_no_probe = pickle.load(csr_file)

    with open("../structures/csr_utility_matrix.pkl", "rb") as csr_file:
        csr_utility_matrix = pickle.load(csr_file)

    with open("../structures/user_biases_no_probe.pkl", "rb") as user_biases_no_probe_file:
        user_biases_no_probe = pickle.load(user_biases_no_probe_file)

    with open("../structures/movie_biases_no_probe.pkl", "rb") as movie_biases_no_probe_file:
        movie_biases_no_probe = pickle.load(movie_biases_no_probe_file)

    global_avg = csr_no_probe.data.mean()

    movie_biases_no_probe = np.array(movie_biases_no_probe)
    user_biases_no_probe = np.array(user_biases_no_probe)

    _, probe_movies_rated_by_user, _ = extract_probe("../structures/mappings.pkl", "../txt_data/probe.txt")


    print("Loading parameters...")
    with open("parameters/weights.pkl", "rb") as weights_file:
        weights = pickle.load(weights_file)

    with open("parameters/implicit_weights.pkl", "rb") as implicit_weights_file:
        implicit_weights = pickle.load(implicit_weights_file)

    with open("parameters/trained_movie_biases.pkl", "rb") as trained_movie_biases_file:
        trained_movie_biases = pickle.load(trained_movie_biases_file)

    with open("parameters/trained_user_biases.pkl", "rb") as trained_user_biases_file:
        trained_user_biases = pickle.load(trained_user_biases_file)



    test_global_ngbr()

