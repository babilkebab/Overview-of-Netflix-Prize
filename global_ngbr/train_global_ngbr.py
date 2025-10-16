import pickle
import sys
import numpy as np
from sklearn.metrics import root_mean_squared_error
import psutil, os

p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)

def train_global_ngbr(epochs=10, lr=5e-3, _lambda=5e-3, clip_norm=2.0):
    global csr_no_probe, weights, trained_movie_biases, trained_user_biases, global_avg, implicit_weights, movie_biases_no_probe, user_biases_no_probe

    rmse = 0

    for epoch in range(epochs):
        elapsed = 0
        pred_ratings = np.zeros(100480507-1408395)

        old_rmse = rmse

        for user in range(480189):
            user_row = csr_no_probe[user]
            user_rated_movies = user_row.indices
            if user_rated_movies.size == 0:
                continue
            true_ratings = user_row.data
            user_bias_on_all_movies = global_avg + movie_biases_no_probe[user_rated_movies] + user_biases_no_probe[user]
            centered_ratings = true_ratings - user_bias_on_all_movies
            inv_sqrt = 1.0 / np.sqrt(len(user_rated_movies))
            for i, movie in enumerate(user_rated_movies):

                bias_user_on_movie = global_avg + trained_user_biases[user] + trained_movie_biases[movie]
                weights[movie, movie] = 0.0
                implicit_weights[movie, movie] = 0.0


                pred_rating = bias_user_on_movie + inv_sqrt * (weights[movie, user_rated_movies].dot(centered_ratings) + implicit_weights[movie, user_rated_movies].sum())
                pred_ratings[elapsed] = pred_rating
                err = pred_rating - true_ratings[i]

                grad_vec_w = inv_sqrt * 2.0 * err * centered_ratings + 2 * _lambda * weights[movie, user_rated_movies]
                grad_vec_w[i] = 0.0

                grad_vec_c = inv_sqrt * 2.0 * err + 2 * _lambda * implicit_weights[movie, user_rated_movies]
                grad_vec_c[i] = 0.0

                grad_user_bias =  2.0 * err + 2 * _lambda * trained_user_biases[user]
                grad_movie_bias =  2.0 * err + 2 * _lambda * trained_movie_biases[movie]

                
                norm_w = np.linalg.norm(grad_vec_w)
                if norm_w > clip_norm:
                    grad_vec_w *= clip_norm / norm_w

                norm_c = np.linalg.norm(grad_vec_c)
                if norm_c > clip_norm:
                    grad_vec_c *= clip_norm / norm_c

                norm_ub = np.linalg.norm(grad_user_bias)
                if norm_ub > clip_norm:
                    grad_user_bias *= clip_norm / norm_ub

                norm_mb = np.linalg.norm(grad_movie_bias)
                if norm_mb > clip_norm:
                    grad_movie_bias *= clip_norm / norm_mb


                weights[movie, user_rated_movies] -= lr * grad_vec_w
                weights[movie, movie] = 0.0

                implicit_weights[movie, user_rated_movies] -= lr * grad_vec_c
                implicit_weights[movie, movie] = 0.0

                trained_user_biases[user] -= lr * grad_user_bias
                trained_movie_biases[movie] -= lr * grad_movie_bias

                elapsed+=1
                if elapsed % 100000 == 0:
                    print(f"\rElapsed {(elapsed/(100480507-1408395))*100}% ratings", end="", flush=True)

        all_true_ratings = csr_no_probe.data
        rmse = root_mean_squared_error(all_true_ratings, pred_ratings)
        print(f"\nRMSE at epoch {epoch+1}: {rmse}\n")
        if epoch > 0:
            if abs(rmse - old_rmse) < 1e-3:
                break



EPOCHS = 20


if __name__ == "__main__":

    print("Loading structures...")
    with open("../structures/csr_no_probe.pkl", "rb") as csr_file:
        csr_no_probe = pickle.load(csr_file)

    with open("../structures/user_biases_no_probe.pkl", "rb") as user_biases_no_probe_file:
        user_biases_no_probe = pickle.load(user_biases_no_probe_file)

    with open("../structures/movie_biases_no_probe.pkl", "rb") as movie_biases_no_probe_file:
        movie_biases_no_probe = pickle.load(movie_biases_no_probe_file)

    global_avg = csr_no_probe.data.mean()

    movie_biases_no_probe = np.array(movie_biases_no_probe)
    user_biases_no_probe = np.array(user_biases_no_probe)



    if len(sys.argv) > 1:
        with open("parameters/weights.pkl", "rb") as weights_file:
            weights = pickle.load(weights_file)

        with open("parameters/implicit_weights.pkl", "rb") as implicit_weights_file:
            implicit_weights = pickle.load(implicit_weights_file)

        with open("parameters/trained_movie_biases.pkl", "rb") as trained_movie_biases_file:
            trained_movie_biases = pickle.load(trained_movie_biases_file)

        with open("parameters/trained_user_biases.pkl", "rb") as trained_user_biases_file:
            trained_user_biases = pickle.load(trained_user_biases_file)
    else:
        trained_movie_biases = np.array(movie_biases_no_probe)
        trained_user_biases = np.array(user_biases_no_probe)

        weights = np.random.uniform(0, 0.01, (17770, 17770)).astype(np.float32)
        implicit_weights = np.random.uniform(0, 0.01, (17770, 17770)).astype(np.float32)

        np.fill_diagonal(weights, 0)
        np.fill_diagonal(implicit_weights, 0)





    train_global_ngbr(EPOCHS, lr=1e-2, _lambda=2e-3)

    with open("weights.pkl", "wb") as weights_file:
        pickle.dump(weights, weights_file)

    with open("implicit_weights.pkl", "wb") as implicit_weights_file:
        pickle.dump(implicit_weights, implicit_weights_file)

    with open("trained_movie_biases.pkl", "wb") as trained_movie_biases_file:
        pickle.dump(trained_movie_biases, trained_movie_biases_file)

    with open("trained_user_biases.pkl", "wb") as trained_user_biases_file:
        pickle.dump(trained_user_biases, trained_user_biases_file)