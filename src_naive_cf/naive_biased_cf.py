
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import weighted_avg, extract_probe
import pickle
from sklearn.metrics import root_mean_squared_error
from time import time


def compute_cf_probe_set_biased(top_k, norm=False):
    global probe_data, csr_utility_matrix
    begin = time()
    real_ratings = []
    predicted_ratings = []

    for movie in probe_data.keys():
        movie_bias = movie_biases_no_probe[movie]
        for user in probe_data[movie]:
            real_ratings.append(csr_utility_matrix[user, movie])
            predicted_ratings.append(norm_itemitem_collaborative_filtering_biased(user, movie, top_k, user_biases_no_probe[user], movie_bias, norm))


    end = time()
    print(f"Elapsed time: {end-begin}")
    print(f"RMSE with norm_itemitem_collaborative_filtering_biased (top {top_k}): {root_mean_squared_error(real_ratings, predicted_ratings)}")


def norm_itemitem_collaborative_filtering_biased(user_id, movie_id, top_k, user_bias, movie_bias, norm=False):
    global norm_all_similarities_movies_no_probe, csr_no_probe, global_avg, movie_biases_no_probe, all_similarities_movies_no_probe

    user_row = csr_no_probe[user_id]

    bias = global_avg + user_bias + movie_bias

    rated_movies = user_row.indices
    rated_ratings = user_row.data

    mask = rated_movies != movie_id
    rated_movies = rated_movies[mask]
    rated_ratings = rated_ratings[mask]

    if norm:
        sims = norm_all_similarities_movies_no_probe[movie_id, rated_movies].tolist()[0]
    else:
        sims = all_similarities_movies_no_probe[movie_id, rated_movies].tolist()[0]

    pairs = list(zip(rated_movies, sims, rated_ratings))

    pairs.sort(key=lambda x: x[1], reverse=True)


    top = pairs[:top_k]
    top_k_biases = [global_avg + user_bias + movie_biases_no_probe[movie[0]] for movie in top]
    top_sims    = [sim    for (_, sim,    _) in top]
    top_ratings = [rating-_bias for _bias, (_, _, rating) in zip(top_k_biases, top)]

    rating = bias + weighted_avg(top_ratings, top_sims)


    if rating < 1:
        rating = 1
    if rating > 5:
        rating = 5

    return rating




if __name__ == "__main__":

    with open("../structures/csr_no_probe.pkl", "rb") as csr_no_probe_file:
        csr_no_probe = pickle.load(csr_no_probe_file)

    with open("../structures/csr_utility_matrix.pkl", "rb") as csr_utility_matrix_file:
        csr_utility_matrix = pickle.load(csr_utility_matrix_file)

    with open("../structures/norm_all_similarities_movies_no_probe.pkl", "rb") as norm_all_similarities_movies_no_probe_file:
        norm_all_similarities_movies_no_probe = pickle.load(norm_all_similarities_movies_no_probe_file)

    with open("../structures/all_similarities_movies_no_probe.pkl", "rb") as all_similarities_movies_no_probe_file:
        all_similarities_movies_no_probe = pickle.load(all_similarities_movies_no_probe_file)

    with open("../structures/user_biases_no_probe.pkl", "rb") as user_biases_no_probe_file:
        user_biases_no_probe = pickle.load(user_biases_no_probe_file)

    with open("../structures/movie_biases_no_probe.pkl", "rb") as movie_biases_no_probe_file:
        movie_biases_no_probe = pickle.load(movie_biases_no_probe_file)



    probe_data, _, _ = extract_probe("../structures/mappings.pkl", "../txt_data/probe.txt")

    global_avg = csr_no_probe.data.mean()

    top_ks = [2,5,10,20,50,100]

    for top_k in top_ks:
        compute_cf_probe_set_biased(top_k, norm=True)

    for top_k in top_ks:
        compute_cf_probe_set_biased(top_k, norm=False)