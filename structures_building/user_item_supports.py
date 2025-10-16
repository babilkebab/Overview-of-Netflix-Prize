import pandas as pd
import pickle 





if __name__ == "__main__":
    print("Loading training data...")
    df_train = pd.read_csv("../structures/df_train_mapped.csv", header=0)

    print("Computing supports...")
    user_supports = df_train.groupby("user")["item"].agg("count")
    item_supports = df_train.groupby("item")["user"].agg("count")

    print("Computing frequencies...")
    user_frequencies = df_train.groupby(["user", "time"]).size().reset_index(name='count')
    user_frequencies_dict = {}
    for _, row in user_frequencies.iterrows():
        user = row['user']
        time = row['time']
        count = row['count']
        
        if user not in user_frequencies_dict:
            user_frequencies_dict[user] = {}
        
        user_frequencies_dict[user][time] = count


    with open("../structures/user_supports.pkl", "wb") as f:
        pickle.dump(user_supports.to_dict(), f)

    with open("../structures/user_frequencies.pkl", "wb") as f:
        pickle.dump(user_frequencies_dict, f)

    with open("../structures/item_supports.pkl", "wb") as f:
        pickle.dump(item_supports.to_dict(), f)