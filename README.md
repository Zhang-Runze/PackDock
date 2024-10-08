# PackDock: a Diffusion Based Side Chain Packing Model for Flexible Protein-Ligand Docking 


This repo contains a PyTorch implementation for the paper  PackDock: a Diffusion Based Side Chain Packing Model for Flexible Protein-Ligand Docking 

If you have any question, feel free to open an issue or reach out to us: [zhangrunze@simm.ac.cn](zhangrunze@simm.ac.cn)✉️.

by [Runze Zhang](https://github.com/Zhang-Runze)
# PackDock Overview
![](https://github.com/Zhang-Runze/PackDock/blob/main/figs/Method%20Overview.jpg)


# Setup Environment

We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

Clone the current repo

    git clone https://github.com/Zhang-Runze/PackDock.git
    
This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):

    conda create --name packdock python=3.8
    conda activate packdock
    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    python -m pip install openbabel PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
    
Then you need to install a ligand conformation sampling algorithm (such as [AutoDock-Vina](https://github.com/ccsb-scripps/AutoDock-Vina), [gnina](https://github.com/gnina/gnina), [Vina-GPU](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0), etc.).
It's worth noting that PackDock offers a highly general flexible docking strategy, capable of integrating any ligand conformation sampling algorithm develop yourself or you might choose to use. This implies that employing more  advanced ligand conformation sampling algorithms could potentially lead to unexpectedly impressive docking results.


# Data preprocess
First run:

    python get_pocket/get_pocket.py
    
Then proceed with further processing using [ADFR-Suite](https://ccsb.scripps.edu/adfr/downloads/) and OpenBabel.


# Model weights

The model weights are available on [zenodo](https://zenodo.org/records/10851699).


# Running PackPocket on your protein

Protein pocket packing:

    python -m evaluate_protein --run_name Protein pocket packing --inference_steps 20 --samples_per_complex 10 --batch_size 10 --actual_steps 20 --cache_path data/cacheTest_CASP14 --no_final_step_noise --data_dir data/CASP/CASP14/ --split_path data/splits/CASP14_list --save_visualisation

Ligand based protein pocket packing:

    python -m evaluate_ligand_based_protein --run_name Ligand based protein pocket packing --inference_steps 20 --samples_per_complex 10 --batch_size 10 --actual_steps 20 --cache_path data/cacheTest_PDBBind --no_final_step_noise --data_dir data/PDBBind_processed --split_path data/splits/timesplit_test_no_rec_overlap --save_visualisation

# Running PackDock on your complex

    python docking_evaluate_gnina_apo.py --rmsd

or

    python docking_evaluate_vina_apo.py --rmsd
    

# Retraining PackPocket
Download the data([BC40](https://zenodo.org/) or [PDBbind](https://zenodo.org/records/6408497)) and place it as described in the "Dataset" section above.

### Training a model yourself and using those weights
Train the PackPocket:

    python -m train_protein --run_name PackPocket_protein --test_sigma_intervals  --log_dir workdir --lr 1e-3 --batch_size 8 --ns 48 --nv 10 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 30.0 --atom_radius 5.0 --cross_distance_embed_dim 64 --distance_embed_dim 64 --sigma_embed_dim 64 --cross_max_distance 20 --num_dataloader_workers 36 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 100 --use_ema --scheduler_patience 30 --n_epochs 300 --all_atoms --num_worker 36 --no_torsion --data_dir data/bc40_pockets_processed/ --split_train data/splits/bc40_train_set --split_val data/splits/bc40_validation_set --split_test data/splits/bc40_test_set 

Train the ligand-based PackPocket:

    python -m train_ligand_based_protein --run_name PackPocket_ligand_based_protein --test_sigma_intervals  --log_dir workdir --lr 1e-3 --batch_size 8 --ns 48 --nv 10 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 30.0 --atom_radius 5.0 --cross_distance_embed_dim 64 --distance_embed_dim 64 --sigma_embed_dim 64 --cross_max_distance 20 --num_dataloader_workers 1 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 100 --use_ema --scheduler_patience 30 --n_epochs 300 --all_atoms --num_worker 36 --no_torsion --data_dir data/PDBBind_processed/ --split_train data/splits/timesplit_no_lig_overlap_train --split_val data/splits/timesplit_no_lig_overlap_val --split_test data/splits/timesplit_test_no_rec_overlap

The model weights are saved in the `workdir` directory.
