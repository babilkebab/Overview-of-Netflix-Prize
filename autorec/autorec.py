import pickle
import math
import torch
from autorec_modules import *
from torch.utils.data import DataLoader



def eval(model, dataloader, device):
    model.eval()

    sse = 0.0
    nobs = 0.0
    processed_cols = 0

    with torch.inference_mode():
        for y, y_test, m in enumerate(dataloader):
            y, y_test, m = y.to(device), y_test.to(device), m.to(device)


            pred = model(y)
            if pred.shape != y.shape:
                pred = pred.T


            pred[m] = pred[m].clamp(min=1.0, max=5.0)

            sse  += ((pred - y_test).pow(2) * m).sum().item()
            nobs += m.sum().item()

            processed_cols += y.shape[1]

    rmse = math.sqrt(sse / max(1.0, nobs))
    print(f"RMSE = {rmse:.6f}")


def train(model, dataloader, device):
    model.train()
    size = len(dataloader.dataset)
    processed_cols = 0
    for batch, (y, m) in enumerate(dataloader):
        y, m = y.to(device), m.to(device)

        pred = model(y).T

        loss = ((pred - y).pow(2) * m).sum() / m.sum().clamp_min(1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        processed_cols += y.shape[1]  
        if batch % 100 == 0:
            print(f"loss: {loss.item():.6f}  [{processed_cols:>5d}/{size:>5d}]")





EPOCHS = 300
BATCH_SIZE=64
PROBE_BATCH_SIZE=64


if __name__ == "__main__":

    
    print("Loading data...")
    with open("../structures/csr_no_probe.pkl", "rb") as csr_file:
        csc_no_probe = (pickle.load(csr_file)).tocsc()
    with open("../structures/csr_utility_matrix.pkl", "rb") as csr_file:
        csc_total = (pickle.load(csr_file)).tocsc()

    dataloader = DataLoader(
        CSCDataset(csc_no_probe),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=make_col_collate(csc_no_probe), pin_memory=True
    )





    n_users = 480189
    hidden = 250
    dropout = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing model...")
    model = AutoRec(n_users=n_users, dim_hidden=hidden).to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if 'bias' in n else decay).append(p)

    optimizer = torch.optim.Adam(
        [{'params': decay, 'weight_decay': 1e-4},
        {'params': no_decay, 'weight_decay': 0.0}],
        lr=1e-3
    )




    mask = (csc_no_probe != 0)

    csc_probe = csc_total - csc_total.multiply(mask)
    csc_probe.eliminate_zeros()
    csr_probe = csc_probe.tocsr()

    probe_dataloader = DataLoader(
        CSCDataset(csc_probe),
        batch_size=PROBE_BATCH_SIZE, shuffle=True,
        collate_fn=make_eval_col_collate(csc_no_probe, csc_probe), pin_memory=True
    )





    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        train(model, dataloader, device)
        if epoch % 10 == 0:
            eval(model, probe_dataloader, device)
