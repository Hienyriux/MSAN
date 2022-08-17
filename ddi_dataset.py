# -*- coding: utf-8 -*-

import json
import random

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data, Batch

class DDIDataset(Dataset):
    def __init__(self, dataset_name, split, fold):
        file_path = f"dataset/{dataset_name}_{split}_{fold}.json"
        
        with open(file_path, "r") as f:
            pairs = json.load(f)
        
        self.pos_pairs = pairs["pos"]
        self.neg_pairs = pairs["neg"]
    
    def __getitem__(self, idx):
        return self.pos_pairs[idx], self.neg_pairs[idx]
    
    def __len__(self):
        return len(self.pos_pairs)
    
    def do_shuffle(self):
        random.shuffle(self.neg_pairs)

class BatchLoader:
    def __init__(self, args):
        self.device = args.device
        
        dataset_name = args.dataset
        dataset_name = dataset_name.lower()
        
        self.graphs = torch.load(f"dataset/{dataset_name}_graphs.pt")
    
    def gen_drug_batch(self, drug_list):
        graph_batch = []
        
        for drug in drug_list:
            graph = self.graphs[drug]
            
            x = graph["x"]
            edge_index = graph["edge_index"]
            data = Data(x, edge_index)
            
            graph_batch.append(data)
        
        return graph_batch

    def proc_batch(self, batch):
        drug_1, drug_2, ddi_type = zip(*batch)
        
        graph_batch_1 = self.gen_drug_batch(drug_1)
        graph_batch_2 = self.gen_drug_batch(drug_2)
        
        return {
            "graph_batch_1" : graph_batch_1,
            "graph_batch_2" : graph_batch_2,
            "ddi_type" : ddi_type
        }

    def collate_fn(self, batch):
        pos_batch, neg_batch = zip(*batch)
        
        ret_pos = self.proc_batch(pos_batch)
        ret_neg = self.proc_batch(neg_batch)
        
        graph_batch_1 = ret_pos["graph_batch_1"] + ret_neg["graph_batch_1"]
        graph_batch_2 = ret_pos["graph_batch_2"] + ret_neg["graph_batch_2"]
        
        y_true = [1] * len(ret_pos["ddi_type"]) + [0] * len(ret_neg["ddi_type"])
        ddi_type = ret_pos["ddi_type"] + ret_neg["ddi_type"]
        
        graph_batch_1 = Batch.from_data_list(graph_batch_1).to(self.device)
        graph_batch_2 = Batch.from_data_list(graph_batch_2).to(self.device)
        
        return graph_batch_1, graph_batch_2, ddi_type, y_true
