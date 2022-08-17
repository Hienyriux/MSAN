# -*- coding: utf-8 -*-

import json
import random

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data, Batch

class DDIDataset(Dataset):
    def __init__(self, dataset_name, split):
        file_path = f"dataset/{dataset_name}_{split}.json"
        
        with open(file_path, "r") as f:
            self.pairs = json.load(f)
    
    def __getitem__(self, idx):
        return self.pairs[idx]
    
    def __len__(self):
        return len(self.pairs)
    
    def do_shuffle(self):
        pass

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
    
    def collate_fn(self, batch):
        drug_1, drug_2, y_true = zip(*batch)
        
        batch_1 = self.gen_drug_batch(drug_1)
        batch_2 = self.gen_drug_batch(drug_2)
        
        batch_1 = Batch.from_data_list(batch_1).to(self.device)
        batch_2 = Batch.from_data_list(batch_2).to(self.device)
        
        return batch_1, batch_2, None, y_true
