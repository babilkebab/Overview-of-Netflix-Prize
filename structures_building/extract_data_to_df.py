import re
import gc
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

timestamp_func = lambda day: int(datetime.strptime(day, "%Y-%m-%d").timestamp())

if __name__ == "__main__":
    #Extract probe set in pairs (u, i) format
    print("Extracting probe data...")
    probe_pairs = []


    with open(f"../txt_data/probe.txt", 'r') as f:
        for line in f:
            if ":" in line:
                movie = int(re.search("[0-9]+", line).group())
            else:
                rating_data = line.split(",")
                user = int(rating_data[0])
                probe_pairs.append((movie, user))

    probe_pairs = set(probe_pairs)




    #Extract training set in tuples (u, i, r, timestamp) format and add ratings to probe

    TRAIN_SIZE = 100480507-1408395
    PROBE_SIZE = 1408395

    users_train = np.zeros(TRAIN_SIZE).astype(np.int32)
    items_train = np.zeros(TRAIN_SIZE).astype(np.int32)
    ratings_train = np.zeros(TRAIN_SIZE).astype(np.int32)
    timestamps_train = np.zeros(TRAIN_SIZE).astype(np.int32)


    users_probe = np.zeros(PROBE_SIZE).astype(np.int32)
    items_probe = np.zeros(PROBE_SIZE).astype(np.int32)
    ratings_probe = np.zeros(PROBE_SIZE).astype(np.int32)
    timestamps_probe = np.zeros(PROBE_SIZE).astype(np.int32)


    cnt_train = 0
    cnt_probe = 0


    print("Extracting train data...")
    for i in range(1,5):
        print(f"FILE {i}")
        with open(f"../txt_data/combined_data_{i}.txt") as f:
            for line in f:
                if "," not in line:
                    movie = int(re.search("[0-9]+", line).group())
                else:
                    rating_data = line.split(",")
                    user = int(rating_data[0])
                    rating = int(rating_data[1])
                    timestamp = timestamp_func(rating_data[2].replace("\n", ""))
                    if (movie, user) in probe_pairs:
                        users_probe[cnt_probe] = user
                        items_probe[cnt_probe] = movie
                        ratings_probe[cnt_probe] = rating
                        timestamps_probe[cnt_probe] = timestamp
                        cnt_probe += 1
                    else:
                        users_train[cnt_train] = user
                        items_train[cnt_train] = movie
                        ratings_train[cnt_train] = rating
                        timestamps_train[cnt_train] = timestamp
                        cnt_train += 1
                del line
        gc.collect()



    #To pandas DataFrame

    dataset_train = pd.DataFrame({"user": users_train, "item": items_train, "label": ratings_train, "time": timestamps_train})
    dataset_probe = pd.DataFrame({"user": users_probe, "item": items_probe, "label": ratings_probe, "time": timestamps_probe})



    #Timestamps to days
    print("Timestamps and mappings...")
    days = list(set(timestamps_train).union(set(timestamps_probe)))
    day_one = min(days) / 86400
    first_timestamp = min(days)
    _in_days_range = ((max(days) / 86400) - day_one) / 30

    dataset_train["time"] = (dataset_train["time"]- first_timestamp) / 86400
    dataset_probe["time"] = (dataset_probe["time"]- first_timestamp) / 86400
    dataset_train["time"] = dataset_train["time"].astype("int16")
    dataset_probe["time"] = dataset_probe["time"].astype("int16")
    dataset_train["bin"] = dataset_train["time"] // _in_days_range
    dataset_probe["bin"] = dataset_probe["time"] // _in_days_range

    dataset_train.loc[dataset_train["bin"] == 30, ["bin"]] = 29
    dataset_probe.loc[dataset_probe["bin"] == 30, ["bin"]] = 29
    dataset_train["bin"] = dataset_train["bin"].astype("int8")
    dataset_probe["bin"] = dataset_probe["bin"].astype("int8")

    
    
    #Mapping user and item IDs
    with open("../structures/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)


    dataset_train["user"] = dataset_train["user"].map(mappings)
    dataset_probe["user"] = dataset_probe["user"].map(mappings)
    
    dataset_train["item"] = dataset_train["item"] - 1
    dataset_probe["item"] = dataset_probe["item"] - 1

    dataset_train.to_csv("../structures/df_train_mapped.csv", index=False)
    dataset_probe.to_csv("../structures/df_probe_mapped.csv", index=False)