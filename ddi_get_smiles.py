# -*- coding: utf-8 -*-

import json

def get_drug_dict_smiles_drugbank():
    id_to_smiles = {}
    
    with open("dataset/raw_data/drugbank.txt", "r") as f:
        lines = f.readlines()[1 : ]
    
    for line in lines:
        parts = line[ : -1].split("\t")
        parts = [part.replace("\"", "").strip() for part in parts]
        id_1, id_2, ddi_type, _, smiles_1, smiles_2 = parts
        id_to_smiles[id_1] = smiles_1
        id_to_smiles[id_2] = smiles_2
        
    id_to_smiles = sorted(list(id_to_smiles.items()))
    drug_dict, smiles_list = zip(*id_to_smiles)
    drug_dict = {item : i for i, item in enumerate(drug_dict)}
    smiles_list = list(smiles_list)
    
    with open("dataset/drugbank_drug_dict.json", "w") as f:
        json.dump(drug_dict, f, indent=4)
    
    with open("dataset/drugbank_smiles.json", "w") as f:
        json.dump(smiles_list, f)

def get_drug_dict_smiles_miracle(dataset_name):
    id_to_smiles = {}
    
    prefix = f"dataset/raw_data/{dataset_name}"
    
    with open(f"{prefix}_drug_list.csv", "r") as f:
        lines = f.readlines()[1 : ]
    
    for line in lines:
        parts = line[ : -1].split(",")
        parts = [part.replace("\"", "").strip() for part in parts]
        
        if dataset_name != "zhang":
            drug_id, smiles = parts
        else:
            _, _, drug_id, smiles = parts
        
        id_to_smiles[drug_id] = smiles
    
    smiles_to_id = {value : key for key, value in id_to_smiles.items()}
    
    with open(f"{prefix}_train.csv", "r") as f:
        lines = f.readlines()[1 : ]
    
    with open(f"{prefix}_valid.csv", "r") as f:
        lines += f.readlines()[1 : ]
    
    with open(f"{prefix}_test.csv", "r") as f:
        lines += f.readlines()[1 : ]
    
    drug_dict = set()
    
    for line in lines:
        parts = line[ : -1].split(",")
        parts = [part.replace("\"", "").strip() for part in parts]
        
        if dataset_name == "zhang":
            drug_1, drug_2, smiles_2, smiles_1, _, _, label = parts
        elif dataset_name == "miner":
            drug_1, drug_2, smiles_1, smiles_2, label = parts
        else:
            smiles_1, smiles_2, label = parts
            drug_1 = smiles_to_id.get(smiles_1)
            drug_2 = smiles_to_id.get(smiles_2)
            assert (drug_1 is not None) and (drug_2 is not None)
        
        if dataset_name != "deep":
            assert (drug_1 in id_to_smiles) and (drug_2 in id_to_smiles)
        
        drug_dict.add(drug_1)
        drug_dict.add(drug_2)
    
    drug_dict = sorted(list(drug_dict))
    smiles_list = [id_to_smiles[drug_id] for drug_id in drug_dict]
    drug_dict = {item : i for i, item in enumerate(drug_dict)}
    
    with open(f"dataset/{dataset_name}_drug_dict.json", "w") as f:
        json.dump(drug_dict, f, indent=4)
    
    with open(f"dataset/{dataset_name}_smiles.json", "w") as f:
        json.dump(smiles_list, f)

##get_drug_dict_smiles_drugbank()
##get_drug_dict_smiles_miracle("zhang")
##get_drug_dict_smiles_miracle("miner")
##get_drug_dict_smiles_miracle("deep")
