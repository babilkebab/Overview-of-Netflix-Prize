import numpy as np
import pickle

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cosine_similarities


if __name__ == "__main__":

    #CENTERED RATINGS
    print("Centering ratings...")
    with open("../structures/csr_no_probe.pkl", "rb") as csr_no_probe_file:
        csr_no_probe = pickle.load(csr_no_probe_file)

    normalized_user_avg_csr_no_probe = csr_no_probe.copy()
    normalized_user_avg_csr_no_probe = normalized_user_avg_csr_no_probe.astype(np.float32)

    for i in range(480189): #N_users
        user_rating_avg = normalized_user_avg_csr_no_probe[i].data.mean()

        s = normalized_user_avg_csr_no_probe.indptr[i]
        e = normalized_user_avg_csr_no_probe.indptr[i + 1]

        normalized_user_avg_csr_no_probe.data[s:e] -= user_rating_avg

    normalized_user_avg_csr_no_probe.eliminate_zeros()

    #NORMALIZED SIMILARITIES
    print("Computing similarities...")
    norm_all_similarities_movies_no_probe = cosine_similarities(normalized_user_avg_csr_no_probe)
    norm_all_similarities_movies_no_probe = norm_all_similarities_movies_no_probe.todense().astype(np.float32)

    with open("../structures/norm_all_similarities_movies_no_probe.pkl", "wb") as norm_sim_mtx_no_probe_file:
        pickle.dump(norm_all_similarities_movies_no_probe, norm_sim_mtx_no_probe_file)