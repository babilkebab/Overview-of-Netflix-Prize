from torch import nn
import torch
import torch.nn.functional as F


class FunkSVD(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int):
        super().__init__()
        self.P = nn.Embedding(n_users, factors)
        self.Q = nn.Embedding(n_items, factors)

        nn.init.normal_(self.P.weight, 0, 0.1)
        nn.init.normal_(self.Q.weight, 0, 0.1)


    def forward(self, x):
        user, item = x[:, 0], x[:, 1]

        pred = (self.Q(item) * self.P(user)).sum(axis=1)

        return pred
    




class BaseBiasedSVD(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, training_global_avg: float):
        super().__init__()
        self.P = nn.Embedding(n_users, factors)
        self.Q = nn.Embedding(n_items, factors)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)

        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)
        nn.init.normal_(self.P.weight, 0, 0.1)
        nn.init.normal_(self.Q.weight, 0, 0.1)

        self.register_buffer("mu", torch.tensor(float(training_global_avg)))


    def forward(self, x):
        user, item = x[:, 0], x[:, 1]

        bu = self.bu(user).squeeze(-1)
        bi = self.bi(item).squeeze(-1)
        qi = self.Q(item)
        pu = self.P(user)

        pred = self.mu + bu + bi + (qi * pu).sum(axis=1)

        return pred





class BaseBiasedSVDpp(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, training_global_avg: float, user_rated_movies: dict,  user_inv_sqrt: dict):
        super().__init__()

        self.P  = nn.Embedding(n_users, factors)   
        self.Q  = nn.Embedding(n_items, factors)   
        self.bu = nn.Embedding(n_users, 1)         
        self.bi = nn.Embedding(n_items, 1)         

        self.Ybag = nn.EmbeddingBag(
            num_embeddings=n_items,
            embedding_dim=factors,
            mode='sum',
            include_last_offset=False
        )

        nn.init.normal_(self.P.weight, 0.0, 0.1)
        nn.init.normal_(self.Q.weight, 0.0, 0.1)
        nn.init.normal_(self.Ybag.weight, 0.0, 0.01)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)

        self.register_buffer("mu", torch.tensor(float(training_global_avg), dtype=torch.float32))

        items_flat_list = []
        offsets = [0]
        lengths = []
        inv_sqrt_list = []

        for u in range(n_users):
            items_u = user_rated_movies.get(u, torch.empty(0, dtype=torch.long))
            if items_u.dtype != torch.long:
                items_u = items_u.long()
            items_flat_list.append(items_u)

            lu = int(items_u.numel())
            lengths.append(lu)
            offsets.append(offsets[-1] + lu)

            inv_val = user_inv_sqrt.get(u, torch.tensor(0.0))
            inv_sqrt_list.append(float(inv_val.item()))

        del user_rated_movies
        del user_inv_sqrt

        items_flat = torch.cat(items_flat_list, dim=0) if len(items_flat_list) else torch.empty(0, dtype=torch.long)
        offsets_t  = torch.tensor(offsets[:-1], dtype=torch.long)         
        lengths_t  = torch.tensor(lengths, dtype=torch.long)              
        inv_sqrt_t = torch.tensor(inv_sqrt_list, dtype=torch.float32)     

        del items_flat_list
        del offsets
        del lengths
        del inv_sqrt_list

        self.register_buffer("items_flat", items_flat)
        self.register_buffer("offsets",    offsets_t)
        self.register_buffer("lengths",    lengths_t)
        self.register_buffer("inv_sqrt_t", inv_sqrt_t)

    def _yj_sum_batch(self, user: torch.Tensor):

        offs_b = self.offsets[user].to(user.device)   
        lens_b = self.lengths[user].to(user.device)   
        total = int(lens_b.sum().item())

        if total == 0:
            return torch.zeros(user.size(0), self.Ybag.embedding_dim, device=user.device)


        idx_list = [self.items_flat[o:o+l] for o, l in zip(offs_b.tolist(), lens_b.tolist())]
        batch_items_flat = torch.cat(idx_list, dim=0).to(user.device)  

        batch_offsets = torch.nn.functional.pad(lens_b.cumsum(dim=0), (1, 0))[:-1]  

        yj_sum = self.Ybag(batch_items_flat, batch_offsets)
        return yj_sum

    def forward(self, x):
        user = x[:, 0].long()
        item = x[:, 1].long()

        pu = self.P(user)                         
        qi = self.Q(item)                         
        bu = self.bu(user).squeeze(-1)            
        bi = self.bi(item).squeeze(-1)            

        y_sum = self._yj_sum_batch(user)                              
        scale = self.inv_sqrt_t[user].to(user.device).unsqueeze(1)    
        pu_tilde = pu + scale * y_sum                                 

        pred = self.mu + bu + bi + (pu_tilde * qi).sum(dim=1)
                 
        return pred


class MiniTimeSVDpp(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, training_global_avg: float, user_rated_movies: dict,  user_inv_sqrt: dict):
        super().__init__()

        self.P  = nn.Embedding(n_users, factors)   
        self.Q  = nn.Embedding(n_items, factors)   
        self.bu = nn.Embedding(n_users, 1)         
        self.bi = nn.Embedding(n_items, 1)         

        self.Ybag = nn.EmbeddingBag(
            num_embeddings=n_items,
            embedding_dim=factors,
            mode='sum',
            include_last_offset=False
        )

        self.bi_bin = nn.Parameter(torch.zeros(n_items, 30))

        self.bu_bin = nn.Parameter(torch.zeros(n_users, 30))

        nn.init.normal_(self.P.weight, 0.0, 0.1)
        nn.init.normal_(self.Q.weight, 0.0, 0.1)
        nn.init.normal_(self.Ybag.weight, 0.0, 0.01)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)

        self.register_buffer("mu", torch.tensor(float(training_global_avg), dtype=torch.float32))

        items_flat_list = []
        offsets = [0]
        lengths = []
        inv_sqrt_list = []

        for u in range(n_users):
            items_u = user_rated_movies.get(u, torch.empty(0, dtype=torch.long))
            if items_u.dtype != torch.long:
                items_u = items_u.long()
            items_flat_list.append(items_u)

            lu = int(items_u.numel())
            lengths.append(lu)
            offsets.append(offsets[-1] + lu)

            inv_val = user_inv_sqrt.get(u, torch.tensor(0.0))
            inv_sqrt_list.append(float(inv_val.item()))

        del user_rated_movies
        del user_inv_sqrt

        items_flat = torch.cat(items_flat_list, dim=0) if len(items_flat_list) else torch.empty(0, dtype=torch.long)
        offsets_t  = torch.tensor(offsets[:-1], dtype=torch.long)         
        lengths_t  = torch.tensor(lengths, dtype=torch.long)              
        inv_sqrt_t = torch.tensor(inv_sqrt_list, dtype=torch.float32)     

        del items_flat_list
        del offsets
        del lengths
        del inv_sqrt_list

        self.register_buffer("items_flat", items_flat)
        self.register_buffer("offsets",    offsets_t)
        self.register_buffer("lengths",    lengths_t)
        self.register_buffer("inv_sqrt_t", inv_sqrt_t)

    def _yj_sum_batch(self, user: torch.Tensor):

        offs_b = self.offsets[user].to(user.device)   
        lens_b = self.lengths[user].to(user.device)   
        total = int(lens_b.sum().item())

        if total == 0:
            return torch.zeros(user.size(0), self.Ybag.embedding_dim, device=user.device)


        idx_list = [self.items_flat[o:o+l] for o, l in zip(offs_b.tolist(), lens_b.tolist())]
        batch_items_flat = torch.cat(idx_list, dim=0).to(user.device)  

        batch_offsets = torch.nn.functional.pad(lens_b.cumsum(dim=0), (1, 0))[:-1]  

        yj_sum = self.Ybag(batch_items_flat, batch_offsets)
        return yj_sum

    def forward(self, x):
        user = x[:, 0].long()
        item = x[:, 1].long()
        bin = x[:, 2].long()
        
        pu = self.P(user)                         
        qi = self.Q(item)                         
        bu = self.bu(user).squeeze(-1)            
        bi = self.bi(item).squeeze(-1)            
        bi_bin = self.bi_bin[item, bin].squeeze(-1)
        bu_bin = self.bu_bin[user, bin].squeeze(-1)

        y_sum = self._yj_sum_batch(user)                              
        scale = self.inv_sqrt_t[user].to(user.device).unsqueeze(1)    
        pu_tilde = pu + scale * y_sum                                 

        pred = self.mu + bu + bi + bi_bin + bu_bin + (pu_tilde * qi).sum(dim=1)
                 
        return pred
