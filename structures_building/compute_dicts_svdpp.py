import pandas as pd
import numpy as np
import torch
import pickle

print("Reading training set...")
df_train = pd.read_csv("../structures/df_train_mapped.csv", header=0)

print("Creating users_rated_movies_dict...")
users_rated_movies = df_train.groupby(by="user")['item'].agg(list)
users_rated_movies_dict = users_rated_movies.to_dict()
users_rated_movies_dict = {k: torch.tensor(v) for k, v in users_rated_movies_dict.items()}

print("Creating users_inv_sqrt_dict...")
users_inv_sqrt = df_train.groupby('user')['item'].agg(lambda x: 1 / np.sqrt(len(x)))
users_inv_sqrt_dict = users_inv_sqrt.to_dict()
users_inv_sqrt_dict = {k: torch.tensor(v) for k, v in users_inv_sqrt_dict.items()}


print("Saving structures...")
with open("../structures/users_rated_movies_dict.pkl", "wb") as f:
    pickle.dump(users_rated_movies_dict, f)

with open("../structures/users_inv_sqrt_dict.pkl", "wb") as f:
    pickle.dump(users_inv_sqrt_dict, f)