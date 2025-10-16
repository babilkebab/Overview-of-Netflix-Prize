import pandas as pd
from pathlib import Path
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error



def ensemble_dataset(path):
    path = Path(path)
    csv_files = list(path.glob("*.csv"))
    result_df = pd.read_csv(csv_files[0], header=0)
    for csv_file in csv_files[1:]:
        try:
            df = pd.read_csv(csv_file, header=0)
            result_df = pd.merge(result_df, df, on=["user", "item", "label", "time", "bin"])
            
        except Exception as e:
            print(f"Error {csv_file.name}: {e}")

    return result_df




if __name__ == "__main__":
    print("Creating the ensemble dataset...")
    ensemble_preds_df_train = ensemble_dataset("training_set")
    ensemble_preds_df_test = ensemble_dataset("test_set")


    #Integrating user and item supports, user daily frequencies like BellKor implementation

    with open("../structures/user_supports.pkl", "rb") as f:
        user_supports = pickle.load(f)

    with open("../structures/user_frequencies.pkl", "rb") as f:
        user_frequencies_dict = pickle.load(f)

    with open("../structures/item_supports.pkl", "rb") as f:
        item_supports = pickle.load(f)


    ensemble_preds_df_train["user_support"] = ensemble_preds_df_train["user"].map(user_supports)
    ensemble_preds_df_train["item_support"] = ensemble_preds_df_train["item"].map(item_supports)

    ensemble_preds_df_test["user_support"] = ensemble_preds_df_test["user"].map(user_supports)
    ensemble_preds_df_test["item_support"] = ensemble_preds_df_test["item"].map(item_supports)

    ensemble_preds_df_train["user_frequency"] = ensemble_preds_df_train.apply(
        lambda row: user_frequencies_dict.get(row["user"], {}).get(row["time"], 0), axis=1
    )

    ensemble_preds_df_test["user_frequency"] = ensemble_preds_df_test.apply(
        lambda row: user_frequencies_dict.get(row["user"], {}).get(row["time"], 0), axis=1
    )


    # GBDT training and eval
    print("Training ensemble model...")
    X_train, X_test, y_train, y_test = ensemble_preds_df_train.drop(columns=["user", "item", "label"]), \
                                        ensemble_preds_df_test.drop(columns=["user", "item", "label"]), \
                                        ensemble_preds_df_train["label"], ensemble_preds_df_test["label"]


    ensemble = XGBRegressor(n_estimators=25, max_depth=5, learning_rate=0.2, subsample=1, objective="reg:squarederror")

    ensemble.fit(X_train, y_train)

    ensemble_preds = ensemble.predict(X_test)

    print(f"RMSE on test set: {root_mean_squared_error(y_test, ensemble_preds)}")