import pickle


if __name__ == "__main__":

    with open("../structures/csr_no_probe.pkl", "rb") as csr_no_probe_file:
        csr_no_probe = pickle.load(csr_no_probe_file)

    csc_no_probe = csr_no_probe.tocsc()


    global_avg = csr_no_probe.data.mean()
    user_biases_no_probe = [(csr_no_probe[i].data.mean() - global_avg) for i in range(csr_no_probe.shape[0])]
    movie_biases_no_probe = [(csc_no_probe.getcol(i).data.mean() - global_avg) for i in range(csc_no_probe.shape[1])]


    with open("../structures/user_biases_no_probe.pkl", "wb") as user_biases_no_probe_file:
        pickle.dump(user_biases_no_probe, user_biases_no_probe_file)

    with open("../structures/movie_biases_no_probe.pkl", "wb") as movie_biases_no_probe_file:
        pickle.dump(movie_biases_no_probe, movie_biases_no_probe_file)