import re
import gc
import numpy as np
import pickle
import scipy.sparse as sp

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_probe



if __name__ == "__main__":
    users = set()
    users_mapping = {}
    user_inc_counter = 0

    rows = []
    columns = []
    ratings = []

    
    #EXTRACTING NETFLIX PRIZE DATA

    for i in range(1,5):
        print(f"Extracting file {i}...")
        with open(f"../txt_data/combined_data_{i}.txt") as f:
            for line in f:
                if "," not in line:
                    movie = int(re.search("[0-9]+", line).group())
                    movie -= 1
                else:
                    rating_data = line.split(",")
                    user = int(rating_data[0])
                    if user not in users:
                        users_mapping[user] = user_inc_counter
                        user_inc_counter += 1
                        users.add(int(rating_data[0]))
                    mapped_user = users_mapping[user]
                    rows.append(mapped_user)
                    columns.append(movie)
                    ratings.append(int(rating_data[1]))
                del line
        gc.collect()


    with open("../structures/mappings.pkl", "wb") as mappings_file:
        pickle.dump(users_mapping, mappings_file)

    #CSR SPARSE MATRIX 
    print("Data to CSR matrix...")
    rows = np.array(rows)
    columns = np.array(columns)
    ratings = np.array(ratings)

    csr_utility_matrix = sp.csr_matrix((ratings, (rows, columns)))
    with open("../structures/csr_utility_matrix.pkl", "wb") as mtx_file:
        pickle.dump(csr_utility_matrix, mtx_file)

    #CSR SPARSE MATRIX WITHOUT PROBE SET

    _, probe_movies_rated_by_user, _ = extract_probe("../structures/mappings.pkl", "../txt_data/probe.txt")

    csr_no_probe = (csr_utility_matrix.copy()).tolil()

    for user in probe_movies_rated_by_user.keys():
        for movie in probe_movies_rated_by_user[user]:
            csr_no_probe[user, movie] = 0

    csr_no_probe = csr_no_probe.tocsr()
    csr_no_probe.eliminate_zeros()

    with open("../structures/csr_no_probe.pkl", "wb") as csr_no_probe_file:
        pickle.dump(csr_no_probe, csr_no_probe_file)
