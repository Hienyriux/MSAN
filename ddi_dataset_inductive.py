# -*- coding: utf-8 -*-

import json
import random

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data, Batch

class DDIDataset(Dataset):
    def __init__(self, dataset_name, split, fold):
        file_path = f"dataset/{dataset_name}_ind_{split}_{fold}.json"
        
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
        self.fold = args.fold
        
        dataset_name = args.dataset
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        
        self.graphs = torch.load(f"dataset/{dataset_name}_graphs.pt")
        self.nearest_old = self.get_nearest_old()
    
    def get_nearest_old(self):
        sim_mat_path = f"dataset/{self.dataset_name}_tanimoto.pt"
        sim_mat = torch.load(sim_mat_path).to(self.device)
        
        old_new_path = f"dataset/{self.dataset_name}_old_new_{self.fold}.json"
        with open(old_new_path, "r") as f:
            old_idx = json.load(f)["old"]
        
        old_idx = torch.LongTensor(old_idx)
        # (num_old_drugs, )
        
        old_drugs = sim_mat[ : , old_idx]
        # (num_drugs, num_old_drugs)
        
        nearest_old = old_idx[old_drugs.argmax(dim=-1)]
        # (num_drugs, )
        
        # max_sim, _ = old_drugs.max(dim=-1)
        # invalid_sub = (max_sim < 0.5)
        # to_self = torch.arange(nearest_old.size(0))
        # nearest_old[invalid_sub] = to_self[invalid_sub]
        
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

    def gen_drug_batch_test(self, drug_list):
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

    def proc_batch_test(self, batch):
        drug_1, drug_2, ddi_type = zip(*batch)
        
        graph_batch_1, graph_batch_old_1 = self.gen_drug_batch_test(drug_1)
        graph_batch_2, graph_batch_old_2 = self.gen_drug_batch_test(drug_2)
        
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
        
        y_true = [1] * len(ret_pos["ddi_type"]) + [0] * len(ret_neg["ddi_type"])
        ddi_type = ret_pos["ddi_type"] + ret_neg["ddi_type"]
        
        graph_batch_1 = Batch.from_data_list(graph_batch_1).to(self.device)
        graph_batch_2 = Batch.from_data_list(graph_batch_2).to(self.device)
        
        return graph_batch_1, graph_batch_2, None, None, ddi_type, y_true
    
    def collate_fn_test(self, batch):
        pos_batch, neg_batch = zip(*batch)
        
        ret_pos = self.proc_batch_test(pos_batch)
        ret_neg = self.proc_batch_test(neg_batch)
        
        graph_batch_1 = ret_pos["graph_batch_1"] + ret_neg["graph_batch_1"]
        graph_batch_2 = ret_pos["graph_batch_2"] + ret_neg["graph_batch_2"]

        graph_batch_old_1 = ret_pos["graph_batch_old_1"] + ret_neg["graph_batch_old_1"]
        graph_batch_old_2 = ret_pos["graph_batch_old_2"] + ret_neg["graph_batch_old_2"]
        
        y_true = [1] * len(ret_pos["ddi_type"]) + [0] * len(ret_neg["ddi_type"])
        ddi_type = ret_pos["ddi_type"] + ret_neg["ddi_type"]
        
        graph_batch_1 = Batch.from_data_list(graph_batch_1).to(self.device)
        graph_batch_2 = Batch.from_data_list(graph_batch_2).to(self.device)

        graph_batch_old_1 = Batch.from_data_list(graph_batch_old_1).to(self.device)
        graph_batch_old_2 = Batch.from_data_list(graph_batch_old_2).to(self.device)
        
        return graph_batch_1, graph_batch_2, graph_batch_old_1, graph_batch_old_2, ddi_type, y_true
