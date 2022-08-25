# MSAN
Molecular Substructure-Aware Network for Drug-Drug Interaction Prediction

## Requirements
* torch==1.9.1
* torch-geometric==2.0.4

## Run
Run by command line, e.g.:
`python ddi_main.py --device cuda --dataset DrugBank --hidden_dim 128 --num_patterns 60 --batch_size 256`
For detailed command line options, see `ddi_main.py`
