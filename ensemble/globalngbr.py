import numpy as np
import pandas as pd
import pickle



def global_ngbr(data_dict, data_dict_ratings):

    global weights, implicit_weights, user_biases_no_probe, movie_biases_no_probe, trained_movie_biases, trained_user_biases, csr_no_probe, global_avg

    GNgbr_preds = []

    sse = 0

    for user in data_dict.keys():
        user_row = csr_no_probe[user]
        user_rated_movies = user_row.indices
        true_ratings = user_row.data
        user_bias_on_all_movies = global_avg + movie_biases_no_probe[user_rated_movies] + user_biases_no_probe[user]
        centered_ratings = true_ratings - user_bias_on_all_movies
        inv_sqrt = 1.0 / np.sqrt(len(user_rated_movies))
        for i, movie in enumerate(data_dict[user]):
            bias_user_on_movie = global_avg + trained_user_biases[user] + trained_movie_biases[movie]

            self_mask = user_rated_movies == movie

            current_weights = weights[movie, user_rated_movies]
            current_implicit = implicit_weights[movie, user_rated_movies]

            current_weights[self_mask] = 0
            current_implicit[self_mask] = 0

            pred_rating = bias_user_on_movie + inv_sqrt * (current_weights.dot(centered_ratings) + current_implicit.sum())
            true_rating = data_dict_ratings[user][i]

            sse+=(pred_rating-true_rating)**2

            GNgbr_preds.append([user, movie, pred_rating, true_rating])

    print(f"RMSE: {np.sqrt(sse/len(GNgbr_preds))}")
    return GNgbr_preds





if __name__ == "__main__":
    print("Loading data...")
    probe_GBDT = pd.read_csv("../structures/df_probe_GBDT.csv", header=0)
    probe_test = pd.read_csv("../structures/df_probe_test.csv", header=0)

    grouped_probe_GBDT = probe_GBDT.groupby(by="user")["item"]
    grouped_probe_GBDT_ratings = probe_GBDT.groupby(by="user")["label"]
    dict_grouped_probe_GBDT = grouped_probe_GBDT.apply(np.array).to_dict()
    dict_grouped_probe_GBDT_ratings = grouped_probe_GBDT_ratings.apply(np.array).to_dict()

    grouped_probe_test = probe_test.groupby(by="user")["item"]
    grouped_probe_test_ratings = probe_test.groupby(by="user")["label"]
    dict_grouped_probe_test = grouped_probe_test.apply(np.array).to_dict()
    dict_grouped_probe_test_ratings = grouped_probe_test_ratings.apply(np.array).to_dict()


    print("Loading parameters...")
    with open("../global_ngbr/parameters/weights.pkl", "rb") as f:
        weights = pickle.load(f)

    with open("../global_ngbr/parameters/implicit_weights.pkl", "rb") as f:
        implicit_weights = pickle.load(f)

    with open("../global_ngbr/parameters/trained_movie_biases.pkl", "rb") as f:
        trained_movie_biases = pickle.load(f)

    with open("../global_ngbr/parameters/trained_user_biases.pkl", "rb") as f:
        trained_user_biases = pickle.load(f)

    with open("../structures/movie_biases_no_probe.pkl", "rb") as f:
        movie_biases_no_probe = pickle.load(f)

    with open("../structures/user_biases_no_probe.pkl", "rb") as f:
        user_biases_no_probe = pickle.load(f)

    with open("../structures/csr_no_probe.pkl", "rb") as f:
        csr_no_probe = pickle.load(f)

    movie_biases_no_probe = np.array(movie_biases_no_probe)
    user_biases_no_probe = np.array(user_biases_no_probe)
    global_avg = 3.603304257811724

    print("Predicting ratings...")
    GNgbr_preds = global_ngbr(dict_grouped_probe_GBDT, dict_grouped_probe_GBDT_ratings)
    GNgbr_test_preds = global_ngbr(dict_grouped_probe_test, dict_grouped_probe_test_ratings)

    GNgbr_preds_df = pd.DataFrame(GNgbr_preds, columns=["user", "item", "GNgbr_pred", "label"])
    GNgbr_preds_df_test = pd.DataFrame(GNgbr_test_preds, columns=["user", "item", "GNgbr_pred", "label"])

    GNgbr_preds_df = GNgbr_preds_df.merge(probe_GBDT, on=["user", "item", "label"])
    GNgbr_preds_df_test = GNgbr_preds_df_test.merge(probe_test, on=["user", "item", "label"])
    
    print("Saving ratings...")
    GNgbr_preds_df.to_csv("training_set/Gngbr_ratings_train.csv", index=False)
    GNgbr_preds_df_test.to_csv("test_set/Gngbr_ratings_test.csv", index=False)
