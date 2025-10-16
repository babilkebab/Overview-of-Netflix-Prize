import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import BaseBiasedSVD
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import train, eval

if __name__ == "__main__":

    BATCH_SIZE=4096
    K=200

    #Loading datasets into dataloaders
    print("Loading datasets into dataloaders...")
    df_train = pd.read_csv("../structures/df_train_mapped.csv", header=0)
    df_probe = pd.read_csv("../structures/df_probe_mapped.csv", header=0)
    
    global_avg = df_train["label"].mean()

    train_tensor_X = torch.tensor(df_train[["user", "item"]].values)
    train_tensor_Y = torch.tensor(df_train["label"].values, dtype=torch.float32)
    probe_tensor_X = torch.tensor(df_probe[["user", "item"]].values)
    probe_tensor_Y = torch.tensor(df_probe["label"].values)

    tensor_dataset = TensorDataset(train_tensor_X, train_tensor_Y)
    tensor_probe_dataset = TensorDataset(probe_tensor_X, probe_tensor_Y)

    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    probe_dataloader = DataLoader(tensor_probe_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model and optimizer definition
    print("Initializing the model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = BaseBiasedSVD(n_items=17770, n_users=480189, factors=K, training_global_avg=global_avg)

    model.to(device)

    lr=0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training

    EPOCHS = 2

    for epoch in range(EPOCHS):
        train(model, dataloader, device, optimizer, BATCH_SIZE)
        eval(model, probe_dataloader, device)

    MODEL_PATH = f"models/svd{K}.pth"
    torch.save(model, MODEL_PATH)
