import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import weighted_avg, extract_probe
import pickle
from sklearn.metrics import root_mean_squared_error
from time import time


def compute_cf_probe_set(top_k):
    global probe_data, csr_utility_matrix

    begin = time()
    real_ratings = []
    predicted_ratings = []

    for movie in probe_data.keys():
        for user in probe_data[movie]:
            real_ratings.append(csr_utility_matrix[user, movie])
            predicted_ratings.append(naive_itemitem_collaborative_filtering(user, movie, top_k))

    end = time()
    print(f"Elapsed time: {end-begin}")
    print(f"RMSE with naive_itemitem_collaborative_filtering (top {top_k}): {root_mean_squared_error(real_ratings, predicted_ratings)}")


def naive_itemitem_collaborative_filtering(user_id, movie_id, top_k):
    global all_similarities_movies_no_probe, csr_no_probe
    
    user_row = csr_no_probe[user_id]

    rated_movies = user_row.indices
    rated_ratings = user_row.data

    mask = rated_movies != movie_id
    rated_movies = rated_movies[mask]
    rated_ratings = rated_ratings[mask]

    sims = all_similarities_movies_no_probe[movie_id, rated_movies].tolist()[0]

    pairs = list(zip(rated_movies, sims, rated_ratings))

    pairs.sort(key=lambda x: x[1], reverse=True)

    top = pairs[:top_k]
    top_sims    = [sim    for (_, sim,    _) in top]
    top_ratings = [rating for (_, _, rating) in top]

    return weighted_avg(top_ratings, top_sims)


if __name__ == "__main__":

    with open("../structures/csr_no_probe.pkl", "rb") as csr_no_probe_file:
        csr_no_probe = pickle.load(csr_no_probe_file)

    with open("../structures/csr_utility_matrix.pkl", "rb") as csr_utility_matrix_file:
        csr_utility_matrix = pickle.load(csr_utility_matrix_file)

    with open("../structures/all_similarities_movies_no_probe.pkl", "rb") as all_similarities_movies_no_probe_file:
        all_similarities_movies_no_probe = pickle.load(all_similarities_movies_no_probe_file)

    probe_data, _, _ = extract_probe("../structures/mappings.pkl", "../txt_data/probe.txt")

    top_ks = [2,5,10,20,50,100]

    for top_k in top_ks:
        compute_cf_probe_set(top_k)