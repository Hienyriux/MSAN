# -*- coding: utf-8 -*-

import sys
import json

import torch

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

def get_mol_fp_vec(mol):
    config = {"mode" : "ECFP", "num_feats" : 1024, "radius" : 3}
    #config = {
    #    "mode" : "RDKFP", "num_feats" : 1024,
    #    "min_path" : 1, "max_path" : 6, "num_hash_bits" : 4
    #}
    
    mode = config["mode"]
    num_feats = config["num_feats"]
    
    if mode == "ECFP":
        radius = config["radius"]
        fp_obj = GetMorganFingerprintAsBitVect(mol, radius, nBits=num_feats)
    
    elif mode == "RDKFP":
        min_path = config["min_path"]
        max_path = config["max_path"]
        num_hash_bits = config["num_hash_bits"]
        fp_obj = Chem.RDKFingerprint(
            mol, minPath=min_path, maxPath=max_path,
            fpSize=num_feats, nBitsPerHash=num_hash_bits
        )
    
    fp_str = fp_obj.ToBitString()
    fp_vec = list(map(int, fp_str))
    fp_vec = torch.LongTensor(fp_vec)
    
    return fp_vec

def get_mol_fp(dataset_name):    
    with open(f"dataset/{dataset_name}_smiles.json", "r") as f:
        smiles_list = json.load(f)
    
    fp_list = []
    
    for smiles in smiles_list:
        if len(smiles) == 0:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        
        fp_vec = get_mol_fp_vec(mol)
        fp_list.append(fp_vec)
    
    fp_list = torch.stack(fp_list).float()
    
    torch.save(fp_list, f"dataset/{dataset_name}_mol_fp.pt")

def get_tanimoto(dataset_name):
    mol_fp = torch.load(f"dataset/{dataset_name}_mol_fp.pt")
    num_mols = mol_fp.size(0)
    
    tanimoto_mat = [[0.0] * num_mols for i in range(num_mols)]
    
    for i in range(num_mols):
        mol_i = mol_fp[i].sum()
        
        for j in range(i, num_mols):
            mol_j = mol_fp[j].sum()
            
            mol_i_j = (mol_fp[i] * mol_fp[j]).sum()
            tanimoto = mol_i_j / (mol_i + mol_j - mol_i_j)
            
            tanimoto_mat[i][j] = tanimoto
            tanimoto_mat[j][i] = tanimoto
        
        sys.stdout.write(f"\r{i} / {num_mols}")
        sys.stdout.flush()
    
    print()
    tanimoto_mat = torch.Tensor(tanimoto_mat)
    
    torch.save(tanimoto_mat, f"dataset/{dataset_name}_tanimoto.pt")

#get_mol_fp("drugbank")
#get_tanimoto("drugbank")
