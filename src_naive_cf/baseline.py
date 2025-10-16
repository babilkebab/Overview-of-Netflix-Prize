from sklearn.metrics import root_mean_squared_error
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_probe


if __name__ == "__main__":

    with open("../structures/csr_no_probe.pkl", "rb") as csr_no_probe_file:
        csr_no_probe = pickle.load(csr_no_probe_file)

    with open("../structures/csr_utility_matrix.pkl", "rb") as csr_utility_matrix_file:
        csr_utility_matrix = pickle.load(csr_utility_matrix_file)

    probe_data, probe_movies_rated_by_user, probe_users = extract_probe("../structures/mappings.pkl", "../txt_data/probe.txt")
    csc_no_probe = csr_no_probe.tocsc()



    # MOVIE MEANS

    real_ratings = []
    avg_movie_predicted_ratings = []

    for movie in probe_data.keys():
        movie_avg_rating = csc_no_probe.getcol(movie).data.mean()
        avg_movie_predicted_ratings.extend([movie_avg_rating] * len(probe_data[movie]))
        for user in probe_data[movie]:
            real_ratings.append(csr_utility_matrix[user, movie])

    print(f"RMSE with movies mean: {root_mean_squared_error(real_ratings, avg_movie_predicted_ratings)}")




    # USER MEANS

    real_ratings = []
    avg_user_predicted_ratings = []


    for user in probe_users:
        user_ratings = csr_no_probe[user]
        user_avg_rating = user_ratings.data.mean()
        user_ratings_w_probe = csr_utility_matrix[user]
        user_ratings_arr = np.squeeze(np.asarray(user_ratings_w_probe.todense()))
        avg_user_predicted_ratings.extend([user_avg_rating] * len(probe_movies_rated_by_user[user]))
        real_ratings.extend(user_ratings_arr[probe_movies_rated_by_user[user]].tolist())



    print(f"RMSE with users mean: {root_mean_squared_error(real_ratings, avg_user_predicted_ratings)}")





    #GLOBAL MEAN

    avg_rating = csr_no_probe.data.mean()

    avg_predicted_ratings = [avg_rating for i in range(len(real_ratings))]
    print(f"RMSE with global mean: {root_mean_squared_error(real_ratings, avg_predicted_ratings)}")