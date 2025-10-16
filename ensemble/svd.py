import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svd.models import *
from utils import eval
import pandas as pd

BATCH_SIZE=4096
MODEL_NAME=sys.argv[1]
sys.modules['models'] = sys.modules['svd.models']


'''
Syntax: python svd.py model_name
'''


def torch_preds_to_df(model, name, dataloader, device):
    model.eval()

    model_preds = []

    torch.backends.cudnn.benchmark = True

    with torch.inference_mode(): 
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(X)

            if pred.ndim > 1 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
            if y.ndim > 1 and y.size(-1) == 1:
                y = y.squeeze(-1)

            pred = pred.clamp(min=1.0, max=5.0)

            model_preds.append(torch.stack([X[:, 0], X[:, 1], pred, y], dim=1))

    model_preds_cat = torch.cat(model_preds, dim=0)
    
    if device == "cuda":
        model_preds_cat = model_preds_cat.cpu()


    model_preds_df = pd.DataFrame(model_preds_cat, columns=['user', 'item', f'{name}_pred', 'label'])
    model_preds_df["user"] = model_preds_df["user"].astype("int32")
    model_preds_df["item"] = model_preds_df["item"].astype("int32")
    model_preds_df["label"] = model_preds_df["label"].astype("int32")

    return model_preds_df




if __name__ == "__main__":
    print("Loading data into dataloader...")
    probe_GBDT = pd.read_csv("../structures/df_probe_GBDT.csv", header=0)
    probe_test = pd.read_csv("../structures/df_probe_test.csv", header=0)


    if "time" in MODEL_NAME:
        probe_GBDT_tensor_X = torch.tensor(probe_GBDT[["user", "item", "bin"]].values)
        probe_GBDT_tensor_Y = torch.tensor(probe_GBDT["label"].values, dtype=torch.float32)

        probe_test_tensor_X = torch.tensor(probe_test[["user", "item", "bin"]].values)
        probe_test_tensor_Y = torch.tensor(probe_test["label"].values, dtype=torch.float32)

    else:
        probe_GBDT_tensor_X = torch.tensor(probe_GBDT[["user", "item"]].values)
        probe_GBDT_tensor_Y = torch.tensor(probe_GBDT["label"].values, dtype=torch.float32)

        probe_test_tensor_X = torch.tensor(probe_test[["user", "item"]].values)
        probe_test_tensor_Y = torch.tensor(probe_test["label"].values, dtype=torch.float32)


    probe_GBDT_tensor_dataset = TensorDataset(probe_GBDT_tensor_X, probe_GBDT_tensor_Y)
    probe_test_tensor_dataset = TensorDataset(probe_test_tensor_X, probe_test_tensor_Y)


    GBDT_dataloader = DataLoader(probe_GBDT_tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(probe_test_tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    print("Predicting ratings...")
    model = torch.load(f"../svd/models/{MODEL_NAME}.pth", weights_only=False)
    model_df_train = torch_preds_to_df(model, MODEL_NAME, GBDT_dataloader, device)
    model_df_test = torch_preds_to_df(model, MODEL_NAME, test_dataloader, device)

    model_df_train = model_df_train.merge(probe_GBDT, on=["user", "item", "label"])
    model_df_test = model_df_test.merge(probe_test, on=["user", "item", "label"])

    print("Saving ratings...")
    model_df_train.to_csv(f"training_set/{MODEL_NAME}_ratings_train.csv", index=False)
    model_df_test.to_csv(f"test_set/{MODEL_NAME}_ratings_test.csv", index=False)

    eval(model, test_dataloader, device)
