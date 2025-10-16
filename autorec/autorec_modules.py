import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset


class AutoRec(nn.Module):
    def __init__(self, n_users: int, dim_hidden: int):
        super().__init__()
        self.dim_hidden = dim_hidden

        self.autoenc = nn.Sequential(
            nn.Linear(in_features=n_users, out_features=dim_hidden, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=dim_hidden, out_features=n_users, bias=True),
        )

    def forward(self, x):
        decoded_x = self.autoenc(x.T.contiguous())
        return decoded_x
    



class CSCDataset(Dataset):
    def __init__(self, X_csc):
        self.X = X_csc
        self.n_users, self.n_items = X_csc.shape
    
    def __len__(self): 
        return self.n_items
    
    def __getitem__(self, j):
        return int(j)



def make_col_collate(X_csc, rows=None, dtype=np.float32):
    rows = np.arange(X_csc.shape[0], dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    def collate(col_ids):
        cols = np.asarray(col_ids, dtype=np.int64)
        sub = X_csc[rows[:, None], cols]    
                               
        y = torch.from_numpy(sub.toarray().astype(dtype, copy=False))
        m = torch.from_numpy((sub != 0).toarray()).to(torch.bool)
        return y, m
    return collate



def make_eval_col_collate(X_train_csc, X_test_csc, dtype=np.float32):
    rows = np.arange(X_train_csc.shape[0], dtype=np.int64)
    def collate(col_ids):
        cols = np.asarray(col_ids, dtype=np.int64)

        sub_tr = X_train_csc[rows[:, None], cols]
        y_in   = torch.from_numpy(sub_tr.toarray().astype(dtype, copy=False))     

        sub_te = X_test_csc[rows[:, None], cols]
        y_te   = torch.from_numpy(sub_te.toarray().astype(dtype, copy=False))     
        m_te   = torch.from_numpy((sub_te != 0).toarray()).to(torch.bool)        

        return y_in, y_te, m_te
    return collate