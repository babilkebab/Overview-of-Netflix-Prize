import sklearn.preprocessing as pp
import pickle
import torch
from torch import nn
from math import sqrt

def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat

def weighted_avg(ratings: list, similarities: list):
    sum_ratings = 0
    sum_similarities = 0
    for i, rating in enumerate(ratings):
        sum_ratings += (similarities[i] * rating)
        sum_similarities += similarities[i]
    return sum_ratings/sum_similarities


def extract_probe(MAPPINGS_PATH, PROBE_PATH):
    with open(MAPPINGS_PATH, "rb") as mappings_file:
        users_mapping = pickle.load(mappings_file)

    probe_data = {} #movie_id: MAPPED user_id

    probe_users = set()
    probe_movies_rated_by_user = {} #user_id: movie_id s of movies rated by user_id

    with open(PROBE_PATH, "r") as probe_f:
        for line in probe_f:
            if ":" in line:
                movie = (int(line[:-2]))-1
                probe_data[movie] = []
            else:
                user = users_mapping[int(line[:-1])]
                if user not in probe_movies_rated_by_user.keys():
                    probe_movies_rated_by_user[user] = []
                probe_data[movie].append(user)
                probe_users.add(user)
                probe_movies_rated_by_user[user].append(movie)

    return probe_data, probe_movies_rated_by_user, probe_users


def train(model, dataloader, device, optimizer, BATCH_SIZE):
    sse = 0.0  # sum of squared errors
    n = 0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = nn.MSELoss(reduction="mean")(pred, y)

        diff = pred - y
        sse += torch.sum(diff * diff).item()
        n += diff.numel()
        

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def eval(model, dataloader, device):
    model.eval()
    sse = 0.0  # sum of squared errors
    n = 0

    torch.backends.cudnn.benchmark = True

    with torch.inference_mode():  
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            if pred.ndim > 1 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
            if y.ndim > 1 and y.size(-1) == 1:
                y = y.squeeze(-1)

            pred = pred.clamp(min=1.0, max=5.0)

            diff = pred - y

            sse += torch.sum(diff * diff).item()  
            n += diff.numel()


    rmse = sqrt(sse / n)
    print(f"RMSE: {rmse:.6f}")
    return rmse