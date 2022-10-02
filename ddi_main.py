# -*- coding: utf-8 -*-

import argparse

import random
import numpy as np

import torch

from ddi_predictor import InteractionPredictor
from ddi_train import train as train_tranductive
from ddi_train_inductive import train as train_inductive

dataset_to_abbr = {
    "drugbank" : "drugbank",
    "zhangddi" : "zhang",
    "chch-miner" : "miner",
    "deepddi" : "deep"
}

num_node_feats_dict = {"drugbank" : 75, "zhang" : 41, "miner" : 52, "deep" : 77}
num_ddi_types_dict = {"drugbank" : 86, "zhang" : 0, "miner" : 0, "deep" : 0}

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--dataset", type=str, choices=[
            "DrugBank", "ZhangDDI", "ChCh-Miner", "DeepDDI"
        ], default="DrugBank"
    )
    
    parser.add_argument("--inductive", action="store_true", default=False)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2], default=0)
    
    parser.add_argument(
        "--gnn_model", type=str,
        choices=["GCN", "GAT", "GIN"], default="GIN"
    )
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_num_layers", type=int, default=3)
    
    parser.add_argument("--gat_num_heads", type=int, default=8)
    parser.add_argument("--gat_to_concat", action="store_true", default=False)
    
    parser.add_argument("--gin_nn_layers", type=int, default=5)
    
    parser.add_argument("--num_patterns", type=int, default=60)
    parser.add_argument("--attn_out_residual", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pred_mlp_layers", type=int, default=3)
    
    parser.add_argument(
        "--sub_drop_freq", type=str,
        choices=["half", "always", "never"], default="half"
    )
    parser.add_argument(
        "--sub_drop_mode", type=str, choices=[
            "rand_per_graph", "rand_per_batch",
            "biggest", "smallest"
        ], default="rand_per_graph"
    )

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    args.dataset = dataset_to_abbr[args.dataset.lower()]
    args.num_node_feats = num_node_feats_dict[args.dataset]
    args.num_ddi_types = num_ddi_types_dict[args.dataset]
    
    return args

if __name__ == "__main__":
    args = main()
    
    #model.load_state_dict(torch.load("model/model_232.pt", map_location=MY_DEVICE))
    
    set_all_seeds(args.seed)

    if args.dataset != "drugbank":
        args.inductive = False
    
    #args.fold = 0
    model = InteractionPredictor(args).to(args.device)
    
    if not args.inductive:
        train_tranductive(model, args)
    else:
        train_inductive(model, args)
    
    """
    if args.dataset != "drugbank":
        set_all_seeds(1)
    
    args.fold = 1
    model = InteractionPredictor(args).to(args.device)
    
    if not args.inductive:
        train_tranductive(model, args)
    else:
        train_inductive(model, args)

    if args.dataset != "drugbank":
        set_all_seeds(2)
    
    args.fold = 2
    model = InteractionPredictor(args).to(args.device)
    
    if not args.inductive:
        train_tranductive(model, args)
    else:
        train_inductive(model, args)
    """
