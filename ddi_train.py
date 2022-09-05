# -*- coding: utf-8 -*-

import sys

import torch
import torch.nn as nn

import numpy as np
from sklearn import metrics

from ddi_dataset import DDIDataset, BatchLoader
from ddi_dataset_wo_type import DDIDataset as DDIDataset_WT
from ddi_dataset_wo_type import BatchLoader as BatchLoader_WT

from torch.utils.data import DataLoader

def calc_metrics(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    
    y_pred_label = (y_pred >= 0.5).astype(np.int32)
    
    acc = metrics.accuracy_score(y_true, y_pred_label)
    auc = metrics.roc_auc_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred_label)
    
    p = metrics.precision_score(y_true, y_pred_label)
    r = metrics.recall_score(y_true, y_pred_label)
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

@torch.no_grad()
def evaluate(model, loader, set_len):
    cur_num = 0
    y_pred_all, y_true_all = [], []
    
    for batch in loader:
        graph_batch_1, graph_batch_2, ddi_type, y_true = batch
        
        y_pred = model.forward_func(graph_batch_1, graph_batch_2, ddi_type)
        
        y_pred_all.append(y_pred.detach().sigmoid().cpu())
        y_true_all.append(torch.LongTensor(y_true))
        
        cur_num += graph_batch_1.num_graphs // 2
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()
    
    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    
    return calc_metrics(y_pred, y_true)

def train(model, args):
    if args.dataset == "drugbank":
        train_set = DDIDataset(args.dataset, "train", args.fold)
        valid_set = DDIDataset(args.dataset, "valid", args.fold)
        test_set = DDIDataset(args.dataset, "test", args.fold)
        batch_loader = BatchLoader(args)
        forward_func = model.forward_transductive
    else:
        train_set = DDIDataset_WT(args.dataset, "train")
        valid_set = DDIDataset_WT(args.dataset, "valid")
        test_set = DDIDataset_WT(args.dataset, "test")
        batch_loader = BatchLoader_WT(args)
        forward_func = model.forward_wo_type
    
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    test_set_len = len(test_set)
        
    
    train_loader = DataLoader(
        train_set, args.batch_size, True,
        collate_fn=batch_loader.collate_fn
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    test_loader = DataLoader(
        test_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (1.0 if epoch < 200 else 0.1),
        last_epoch=args.start_epoch-1
    )
    
    max_valid_acc, max_test_acc = 0.0, 0.0
    
    for epoch in range(args.num_epoch):
        print(f"Epoch: {args.start_epoch + epoch}")
        
        train_loss = 0.0
        cur_num = 0
        y_pred_all, y_true_all = [], []
        train_set.do_shuffle()
        model.drop_rate_list.clear()
        
        model.train()
        for i, batch in enumerate(train_loader):
            graph_batch_1, graph_batch_2, ddi_type, y_true = batch
            y_true = torch.Tensor(y_true).to(args.device)
            
            y_pred = model.forward_func(graph_batch_1, graph_batch_2, ddi_type)
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
            
            if args.dataset == "drugbank":
                cur_num += graph_batch_1.num_graphs // 2
            else:
                cur_num += graph_batch_1.num_graphs
            
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
            f"Train ACC: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}\n"
            f"Train P:   {train_p:.4f}, Train R:   {train_r:.4f}, Train AP: {train_ap:.4f}"
        )
        
        model.eval()
        valid_acc, valid_auc, valid_f1, valid_p, valid_r, valid_ap = \
            evaluate(model, valid_loader, valid_set_len)
        print()
        print(
            f"Valid ACC: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid F1: {valid_f1:.4f}\n"
            f"Valid P:   {valid_p:.4f}, Valid R:   {valid_r:.4f}, Valid AP: {valid_ap:.4f}"
        )
        
        test_acc, test_auc, test_f1, test_p, test_r, test_ap = \
            evaluate(model, test_loader, test_set_len)
        print()
        print(
            f"Test ACC:  {test_acc:.4f}, Test AUC:  {test_auc:.4f}, Test F1:  {test_f1:.4f}\n"
            f"Test P:    {test_p:.4f}, Test R:    {test_r:.4f}, Test AP:  {test_ap:.4f}"
        )
        
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(model.state_dict(), f"model/model_{args.start_epoch + epoch}.pt")
            print(f"BEST VALID IN EPOCH {args.start_epoch + epoch}")
        
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), f"model/model_{args.start_epoch + epoch}.pt")
            print(f"BEST TEST IN EPOCH {args.start_epoch + epoch}")
        
        scheduler.step()
        
        print()
