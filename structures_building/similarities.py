import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cosine_similarities


if __name__ == "__main__":
    with open("../structures/csr_no_probe.pkl", "rb") as csr_no_probe_file:
        csr_no_probe = pickle.load(csr_no_probe_file)
    
    print("Computing similarities...")
    all_similarities_movies_no_probe = cosine_similarities(csr_no_probe)
    all_similarities_movies_no_probe = all_similarities_movies_no_probe.todense().astype(np.float32)


    with open("../structures/all_similarities_movies_no_probe.pkl", "wb") as sim_mtx_file:
        pickle.dump(all_similarities_movies_no_probe, sim_mtx_file)