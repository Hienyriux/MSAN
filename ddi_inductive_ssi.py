# -*- coding: utf-8 -*-

import sys
import json
import random

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_batch

MY_DEVICE = "cuda"

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calc_metrics(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    
    y_pred_label = (y_pred >= 0.5).astype(np.int32)
    
    acc = metrics.accuracy_score(y_true, y_pred_label)
    auc = metrics.roc_auc_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred_label, zero_division=0)
    
    p = metrics.precision_score(y_true, y_pred_label, zero_division=0)
    r = metrics.recall_score(y_true, y_pred_label, zero_division=0)
    ap = metrics.average_precision_score(y_true, y_pred)
    
    return acc, auc, f1, p, r, ap

def get_drop_rate_stats(drop_rate_list):
    drop_rate_stats = {
        "max" : 0.0,
        "min" : 0.0,
        "mean" : 0.0
    }
    
    if len(drop_rate_list) == 0:
        return drop_rate_stats
    
    drop_rate_stats["max"] = max(drop_rate_list)
    drop_rate_stats["min"] = min(drop_rate_list)
    drop_rate_stats["mean"] = sum(drop_rate_list) / len(drop_rate_list)

    return drop_rate_stats

def proc_data_stats():
    with open("dataset/ssi/ssi_data_stats.json", "r") as f:
        data_stats = json.load(f)
    
    drug_2_type_to_drug_1_set = \
        data_stats["drug_2_type_to_drug_1_set"]
    
    drug_1_type_to_drug_2_set = \
        data_stats["drug_1_type_to_drug_2_set"]
    
    drug_2_type_to_drug_1_set = {
        tuple(map(int, key.split(","))) : set(val) \
        for key, val in \
        drug_2_type_to_drug_1_set.items()
    }
    
    drug_1_type_to_drug_2_set = {
        tuple(map(int, key.split(","))) : set(val) \
        for key, val in \
        drug_1_type_to_drug_2_set.items()
    }
    
    data_stats["drug_2_type_to_drug_1_set"] = \
        drug_2_type_to_drug_1_set
    
    data_stats["drug_1_type_to_drug_2_set"] = \
        drug_1_type_to_drug_2_set
    
    return data_stats

class DDIDataset(Dataset):
    def __init__(self, data_stats, split, fold, neg_mode, sample_mode):
        self.split = split
        self.data_stats = data_stats
        self.sample_mode = sample_mode
        
        with open("dataset/ssi/ssi_drug_dict.json", "r") as f:
            drug_dict = json.load(f)
        
        self.num_drugs = len(drug_dict)
        
        old_new_path = (
            "dataset/ssi/"
            f"ssi_old_new_indices_{fold}.json"
        )
        
        with open(old_new_path, "r") as f:
            old_new_idx = json.load(f)
        
        self.drugs_old = set(old_new_idx["old"])
        self.drugs_new = set(old_new_idx["new"])
        
        if neg_mode == "fixed":
            split_path = f"dataset/ssi/ssi_ind_{split}_{fold}_fixed.json"
        else:
            split_path = f"dataset/ssi/ssi_ind_{split}_{fold}.json"
        
        with open(split_path, "r") as f:
            pairs = json.load(f)
        
        if neg_mode == "fixed":
            self.pos_pairs = pairs["pos"]
            self.neg_pairs = pairs["neg"]
        else:
            self.pos_pairs = pairs
            self.neg_pairs = self.sample_neg_wrapper(pairs)
    
    def __getitem__(self, idx):
        return self.pos_pairs[idx], self.neg_pairs[idx]
    
    def __len__(self):
        return len(self.pos_pairs)
    
    def do_shuffle(self):
        random.shuffle(self.neg_pairs)

    def do_sample(self):
        self.neg_pairs = self.sample_neg_wrapper(self.pos_pairs)
    
    def sample_inductive(self, allowable_set, pos_drugs):
        return random.choice(list(allowable_set - pos_drugs))
    
    def sample_neg(self, drug_1, drug_2, ddi_type):
        if self.sample_mode != "strict":
            allowable_set_1 = set(range(self.num_drugs))
            allowable_set_2 = set(range(self.num_drugs))
        
        elif self.split == "train":
            allowable_set_1 = self.drugs_old
            allowable_set_2 = self.drugs_old
        
        elif self.split == "new_new":
            allowable_set_1 = self.drugs_new
            allowable_set_2 = self.drugs_new
        
        else:
            if drug_1 in self.drugs_old:
                allowable_set_2 = self.drugs_new
            else:
                allowable_set_2 = self.drugs_old
            
            if drug_2 in self.drugs_old:
                allowable_set_1 = self.drugs_new
            else:
                allowable_set_1 = self.drugs_old
        
        cnt_1 = self.data_stats["type_to_drug_1_cnt"][ddi_type]
        cnt_2 = self.data_stats["type_to_drug_2_cnt"][ddi_type]
        prob = cnt_2 / (cnt_1 + cnt_2)
        
        if random.random() < prob:
            drug_2_to_drug_1 = \
                self.data_stats["drug_2_type_to_drug_1_set"][(drug_2, ddi_type)]
            return self.sample_inductive(allowable_set_1, drug_2_to_drug_1), 1
        
        drug_1_to_drug_2 = \
            self.data_stats["drug_1_type_to_drug_2_set"][(drug_1, ddi_type)]
        return self.sample_inductive(allowable_set_2, drug_1_to_drug_2), 2
    
    def sample_neg_wrapper(self, pos_pairs):
        neg_pairs = []
        
        for drug_1, drug_2, ddi_type in pos_pairs:
            sampled, flag = self.sample_neg(drug_1, drug_2, ddi_type)
            
            if flag == 1:
                neg_pairs.append([sampled, drug_2, ddi_type])
            else:
                neg_pairs.append([drug_1, sampled, ddi_type])
        
        return neg_pairs

class BatchLoader:
    def __init__(self, fold):
        self.fold = fold
        self.graphs = torch.load("dataset/ssi/ssi_graphs.pt")
        self.nearest_old = self.get_nearest_old()
    
    def get_nearest_old(self):
        sim_mat_path = "dataset/ssi/ssi_tanimoto.pt"
        sim_mat = torch.load(sim_mat_path).to(MY_DEVICE)
        
        old_new_path = (
            "dataset/ssi/"
            f"ssi_old_new_indices_{self.fold}.json"
        )
        
        with open(old_new_path, "r") as f:
            old_new_idx = json.load(f)
        
        old_idx = old_new_idx["old"]
        new_idx = old_new_idx["new"]
        
        old_idx = torch.LongTensor(old_idx)
        old_drugs = sim_mat[ : , old_idx]
        nearest_old = old_idx[old_drugs.argmax(dim=-1)]
        nearest_old = nearest_old.tolist()
        
        return nearest_old
    
    def gen_drug_batch_train(self, drug_list):
        graph_batch = []
        
        for drug in drug_list:
            graph = self.graphs[drug]
            
            x = graph["x"]
            edge_index = graph["edge_index"]
            data = Data(x, edge_index)
            
            graph_batch.append(data)
        
        return graph_batch
    
    def gen_drug_batch_eval(self, drug_list):
        graph_batch = []
        graph_batch_old = []
        
        for drug in drug_list:
            graph = self.graphs[drug]
            graph_old = self.graphs[self.nearest_old[drug]]
            
            x = graph["x"]
            edge_index = graph["edge_index"]
            data = Data(x, edge_index)
            
            x_old = graph_old["x"]
            edge_index_old = graph_old["edge_index"]
            data_old = Data(x_old, edge_index_old)
            
            graph_batch.append(data)
            graph_batch_old.append(data_old)
        
        return graph_batch, graph_batch_old
    
    def proc_batch_train(self, batch):
        drug_1, drug_2, ddi_type = zip(*batch)
        
        graph_batch_1 = self.gen_drug_batch_train(drug_1)
        graph_batch_2 = self.gen_drug_batch_train(drug_2)
        
        return {
            "graph_batch_1" : graph_batch_1,
            "graph_batch_2" : graph_batch_2,
            "ddi_type" : ddi_type
        }
    
    def proc_batch_eval(self, batch):
        drug_1, drug_2, ddi_type = zip(*batch)
        
        graph_batch_1, graph_batch_old_1 = self.gen_drug_batch_eval(drug_1)
        graph_batch_2, graph_batch_old_2 = self.gen_drug_batch_eval(drug_2)
        
        return {
            "graph_batch_1" : graph_batch_1,
            "graph_batch_2" : graph_batch_2,
            "graph_batch_old_1" : graph_batch_old_1,
            "graph_batch_old_2" : graph_batch_old_2,
            "ddi_type" : ddi_type
        }
    
    def collate_fn_train(self, batch):
        pos_batch, neg_batch = zip(*batch)
        
        ret_pos = self.proc_batch_train(pos_batch)
        ret_neg = self.proc_batch_train(neg_batch)
        
        graph_batch_1 = ret_pos["graph_batch_1"] + ret_neg["graph_batch_1"]
        graph_batch_2 = ret_pos["graph_batch_2"] + ret_neg["graph_batch_2"]
        
        y_true = [1] * len(ret_pos["ddi_type"]) + \
            [0] * len(ret_neg["ddi_type"])
        ddi_type = ret_pos["ddi_type"] + ret_neg["ddi_type"]
        
        graph_batch_1 = Batch.from_data_list(graph_batch_1).to(MY_DEVICE)
        graph_batch_2 = Batch.from_data_list(graph_batch_2).to(MY_DEVICE)
        
        return graph_batch_1, graph_batch_2, None, None, ddi_type, y_true
    
    def collate_fn_eval(self, batch):
        pos_batch, neg_batch = zip(*batch)
        
        ret_pos = self.proc_batch_eval(pos_batch)
        ret_neg = self.proc_batch_eval(neg_batch)
        
        graph_batch_1 = ret_pos["graph_batch_1"] + ret_neg["graph_batch_1"]
        graph_batch_2 = ret_pos["graph_batch_2"] + ret_neg["graph_batch_2"]
        
        graph_batch_old_1 = ret_pos["graph_batch_old_1"] + \
            ret_neg["graph_batch_old_1"]
        graph_batch_old_2 = ret_pos["graph_batch_old_2"] + \
            ret_neg["graph_batch_old_2"]
        
        y_true = [1] * len(ret_pos["ddi_type"]) + \
            [0] * len(ret_neg["ddi_type"])
        ddi_type = ret_pos["ddi_type"] + ret_neg["ddi_type"]
        
        graph_batch_1 = Batch.from_data_list(graph_batch_1).to(MY_DEVICE)
        graph_batch_2 = Batch.from_data_list(graph_batch_2).to(MY_DEVICE)
        
        graph_batch_old_1 = Batch.from_data_list(
            graph_batch_old_1
        ).to(MY_DEVICE)
        
        graph_batch_old_2 = Batch.from_data_list(
            graph_batch_old_2
        ).to(MY_DEVICE)
        
        return (
            graph_batch_1,
            graph_batch_2,
            graph_batch_old_1,
            graph_batch_old_2,
            ddi_type, y_true
        )

class MLP(nn.Sequential):
    def __init__(self, hidden_dim, num_layers, dropout=0.5):
        def build_block(input_dim, output_dim):
            return [
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        
        m = build_block(hidden_dim, 2 * hidden_dim)
        for i in range(1, num_layers - 1):
            m += build_block(2 * hidden_dim, 2 * hidden_dim)
        m.append(nn.Linear(2 * hidden_dim, hidden_dim))
        
        super().__init__(*m)

class GINConv(MessagePassing):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.mlp = MLP(hidden_dim, num_layers, 0.0)
    
    def forward(self, x, edge_index, size=None):
        out = self.propagate(edge_index, x=(x, x), size=size)
        out = x + self.mlp(out)
        return out
    
    def message(self, x_j):
        return x_j

class SubExtractor(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()
        
        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        self.residual = residual
    
    def forward(self, x, batch):
        K = self.W_K(x)
        V = self.W_V(x)
        
        K, mask = to_dense_batch(K, batch)
        V, _ = to_dense_batch(V, batch)
        
        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)
        
        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)
        
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
        A = A.softmax(dim=-2)
        
        out = Q + A @ V
        
        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        
        return out, A.detach().argmax(dim=-2), mask

class InteractionPredictor(nn.Module):
    def __init__(
        self, hidden_dim=128, num_patterns=60, pred_mlp_layers=3,
        num_node_feats=77, num_ddi_types=86,
        do_nn_aug=True):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.num_ddi_types = num_ddi_types
        self.do_nn_aug = do_nn_aug
        
        self.node_fc = nn.Linear(num_node_feats, hidden_dim)
        self.gnn = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for i in range(3)
        ])
        
        self.pool = SubExtractor(
            hidden_dim, num_patterns,
            residual=False
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(
                2 * hidden_dim + \
                    num_patterns * num_patterns + \
                    num_ddi_types,
                hidden_dim
            ),
            MLP(hidden_dim, pred_mlp_layers, 0.5),
            nn.Linear(hidden_dim, 1)
        )
        
        self.drop_rate_list = []
    
    def do_gnn(self, x, edge_index):
        for gnn in self.gnn:
            x = gnn(x, edge_index)
        
        return x
    
    @torch.no_grad()
    def get_sub_to_drop(self, sub_cnt):
        drop_prob = torch.rand(
            sub_cnt.size(0), self.num_patterns
        ).to(MY_DEVICE)
        
        drop_prob[sub_cnt == 0] = 0
        
        sub_to_drop = drop_prob.argmax(dim=-1, keepdim=True)
        return sub_to_drop
    
    @torch.no_grad()
    def do_sub_drop(self, x, edge_index, batch):
        num_nodes = x.size(0)
        
        if num_nodes == 1:
            return x
        
        h = self.node_fc(x)
        h = self.do_gnn(h, edge_index)
        
        _, sub_assign, mask = self.pool(h, batch)
        sub_assign[~mask] = self.num_patterns
        sub_one_hot = F.one_hot(sub_assign, self.num_patterns + 1)
        
        sub_cnt = sub_one_hot.sum(dim=1)[ : , : self.num_patterns]
        sub_to_drop = self.get_sub_to_drop(sub_cnt)
        
        drop_mask = (sub_assign == sub_to_drop)
        drop_mask = drop_mask[mask]
        
        x[drop_mask] = 0
        
        drop_rate = drop_mask.sum().item() / drop_mask.size(0)
        
        return x, drop_rate
    
    def encode_graph(self, graph_batch):
        to_drop = False
        
        if self.training:
            to_drop = (random.random() > 0.5)
        
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        if to_drop:
            x, drop_rate = self.do_sub_drop(x, edge_index, batch)
            self.drop_rate_list.append(drop_rate)
        
        x = self.node_fc(x)
        x = self.do_gnn(x, edge_index)
        
        out = global_add_pool(x, batch)
        
        pool, *_ = self.pool(x, batch)
        pool = F.normalize(pool, dim=-1)
        
        return out, pool
    
    def predict(self, out_1, out_2, pool_1, pool_2, ddi_type):        
        sim = pool_1 @ pool_2.transpose(-1, -2)
        sim = sim.flatten(1)
        
        ddi_type = list(ddi_type)
        ddi_type = torch.eye(self.num_ddi_types)[ddi_type].to(MY_DEVICE)
        
        out = torch.cat([out_1, out_2, sim, ddi_type], dim=-1)
        score = self.mlp(out).squeeze(-1)
        
        return score
    
    def forward(
        self, graph_batch_1, graph_batch_2,
        graph_batch_old_1, graph_batch_old_2, ddi_type):
        
        out_1, pool_1 = self.encode_graph(graph_batch_1)
        out_2, pool_2 = self.encode_graph(graph_batch_2)
        
        score = self.predict(out_1, out_2, pool_1, pool_2, ddi_type)

        if self.training or (not self.do_nn_aug):
            return score
        
        old_out_1, old_pool_1 = self.encode_graph(graph_batch_old_1)
        old_out_2, old_pool_2 = self.encode_graph(graph_batch_old_2)
        
        score_old = self.predict(
            old_out_1, old_out_2,
            old_pool_1, old_pool_2,
            ddi_type
        )
        
        score = torch.stack([score, score_old]).sigmoid().mean(dim=0)
        
        return score

@torch.no_grad()
def evaluate(model, loader, set_len):
    cur_num = 0
    y_pred_all, y_true_all = [], []
    
    for batch in loader:
        graph_batch_1, graph_batch_2, \
        graph_batch_old_1, graph_batch_old_2, \
        ddi_type, y_true = batch
        
        y_pred = model(
            graph_batch_1, graph_batch_2,
            graph_batch_old_1, graph_batch_old_2, ddi_type
        )
        
        y_pred_all.append(y_pred.detach().cpu())
        y_true_all.append(torch.LongTensor(y_true))
        
        cur_num += graph_batch_1.num_graphs // 2
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()
    
    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    
    return calc_metrics(y_pred, y_true)

def train(
    model, fold, num_epoch=300,
    batch_size=256, lr=0.001,
    neg_mode="non-fixed",
    sample_mode="non-strict",
    tag="default"):
    
    data_stats = proc_data_stats()
    
    train_set = DDIDataset(data_stats, "train", fold, neg_mode, sample_mode)
    old_new_set = DDIDataset(data_stats, "old_new", fold, neg_mode, sample_mode)
    new_new_set = DDIDataset(data_stats, "new_new", fold, neg_mode, sample_mode)
    
    train_set_len = len(train_set)
    old_new_set_len = len(old_new_set)
    new_new_set_len = len(new_new_set)
    
    batch_loader = BatchLoader(fold)
    
    train_loader = DataLoader(
        train_set, batch_size, True,
        collate_fn=batch_loader.collate_fn_train
    )
    
    old_new_loader = DataLoader(
        old_new_set, batch_size, False,
        collate_fn=batch_loader.collate_fn_eval
    )
    
    new_new_loader = DataLoader(
        new_new_set, batch_size, False,
        collate_fn=batch_loader.collate_fn_eval
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (1.0 if epoch < 200 else 0.1),
        last_epoch=-1
    )
    
    max_old_new_acc, max_new_new_acc = 0.0, 0.0
    
    best_stats_old_new = {
        "epoch" : 0,
        "ACC" : 0.0,
        "AUC" : 0.0,
        "F1" : 0.0,
        "P" : 0.0,
        "R" : 0.0,
        "AP" : 0.0,
    }
    
    best_stats_new_new = {
        "epoch" : 0,
        "ACC" : 0.0,
        "AUC" : 0.0,
        "F1" : 0.0,
        "P" : 0.0,
        "R" : 0.0,
        "AP" : 0.0,
    }
    
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}")
        
        train_loss = 0.0
        cur_num = 0
        y_pred_all, y_true_all = [], []

        if neg_mode != "fixed":
            train_set.do_sample()
            old_new_set.do_sample()
            new_new_set.do_sample()
        else:
            train_set.do_shuffle()
        
        model.drop_rate_list.clear()
        
        model.train()
        for i, batch in enumerate(train_loader):
            graph_batch_1, graph_batch_2, _, _, ddi_type, y_true = batch
            y_true = torch.Tensor(y_true).to(MY_DEVICE)
            
            y_pred = model(
                graph_batch_1, graph_batch_2,
                None, None, ddi_type
            )
            loss = criterion(y_pred, y_true)
            train_loss += loss.item()
            
            y_pred_all.append(y_pred.detach().sigmoid().cpu())
            y_true_all.append(y_true.detach().long().cpu())
            
            dr_stats = get_drop_rate_stats(model.drop_rate_list)
            dr_stats_print = [f"{val:.4f}" for val in dr_stats.values()]
            dr_stats_print = ", ".join(dr_stats_print)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_num += graph_batch_1.num_graphs // 2
            sys.stdout.write(
                f"\r{cur_num} / {train_set_len}, "
                f"{(train_loss / (i + 1)):.6f}, "
                f"{dr_stats_print}"
                "          "
            )
            sys.stdout.flush()
        
        y_pred = torch.cat(y_pred_all)
        y_true = torch.cat(y_true_all)
        train_acc, train_auc, train_f1, train_p, train_r, train_ap = \
            calc_metrics(y_pred, y_true)
        print()
        print(
            f"Train ACC: {train_acc:.4f}, "
            f"Train AUC: {train_auc:.4f}, "
            f"Train F1: {train_f1:.4f}\n"
            f"Train P:   {train_p:.4f}, "
            f"Train R:   {train_r:.4f}, "
            f"Train AP: {train_ap:.4f}"
        )
        
        model.eval()
        old_new_acc, old_new_auc, old_new_f1, old_new_p, old_new_r, old_new_ap = \
            evaluate(model, old_new_loader, old_new_set_len)
        print()
        print(
            f"Old New ACC:  {old_new_acc:.4f}, "
            f"Old New AUC:  {old_new_auc:.4f}, "
            f"Old New F1:  {old_new_f1:.4f}\n"
            f"Old New P:    {old_new_p:.4f}, "
            f"Old New R:    {old_new_r:.4f}, "
            f"Old New AP:  {old_new_ap:.4f}"
        )
        
        new_new_acc, new_new_auc, new_new_f1, new_new_p, new_new_r, new_new_ap = \
            evaluate(model, new_new_loader, new_new_set_len)
        print()
        print(
            f"New New ACC:  {new_new_acc:.4f}, "
            f"New New AUC:  {new_new_auc:.4f}, "
            f"New New F1:  {new_new_f1:.4f}\n"
            f"New New P:    {new_new_p:.4f}, "
            f"New New R:    {new_new_r:.4f}, "
            f"New New AP:  {new_new_ap:.4f}"
        )
        
        if old_new_acc > best_stats_old_new["ACC"]:
            best_stats_old_new["epoch"] = epoch
            best_stats_old_new["ACC"] = old_new_acc
            best_stats_old_new["AUC"] = old_new_auc
            best_stats_old_new["F1"] = old_new_f1
            best_stats_old_new["P"] = old_new_p
            best_stats_old_new["R"] = old_new_r
            best_stats_old_new["AP"] = old_new_ap
            
            print(f"BEST OLD NEW IN EPOCH {epoch}")
            torch.save(model.state_dict(), f"model/{tag}_{fold}_{epoch}.pt")
        
        if new_new_acc > best_stats_new_new["ACC"]:
            best_stats_new_new["epoch"] = epoch
            best_stats_new_new["ACC"] = new_new_acc
            best_stats_new_new["AUC"] = new_new_auc
            best_stats_new_new["F1"] = new_new_f1
            best_stats_new_new["P"] = new_new_p
            best_stats_new_new["R"] = new_new_r
            best_stats_new_new["AP"] = new_new_ap
            
            print(f"BEST NEW NEW IN EPOCH {epoch}")
            torch.save(model.state_dict(), f"model/{tag}_{fold}_{epoch}.pt")
        
        print()
        print("Cur Best: Type/Epoch/ACC/AUC/F1/P/R/AP")
        
        print("Old New/%03d/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f" % (
            best_stats_old_new["epoch"],
            best_stats_old_new["ACC"],
            best_stats_old_new["AUC"],
            best_stats_old_new["F1"],
            best_stats_old_new["P"],
            best_stats_old_new["R"],
            best_stats_old_new["AP"],
        ))
        
        print("New New/%03d/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f" % (
            best_stats_new_new["epoch"],
            best_stats_new_new["ACC"],
            best_stats_new_new["AUC"],
            best_stats_new_new["F1"],
            best_stats_new_new["P"],
            best_stats_new_new["R"],
            best_stats_new_new["AP"],
        ))
        
        scheduler.step()
        
        print()

if __name__ == "__main__":
    set_all_seeds(0)

    fold = 1
    
    model = InteractionPredictor(
        hidden_dim=128,
        num_patterns=60,
        pred_mlp_layers=3,
        num_node_feats=77,
        num_ddi_types=86,
        do_nn_aug=True
    ).to(MY_DEVICE)
    
    train(
        model, fold, num_epoch=300,
        batch_size=256, lr=0.001,
        neg_mode="non-fixed",
        sample_mode="non-strict",
        tag="gcn_nn_nf_ns"
    )
