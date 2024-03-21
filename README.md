# PackDock: a Diffusion Based Side Chain Packing Model for Flexible Protein-Ligand Docking 

Code will be available after our paper has been published!

This repo contains a PyTorch implementation for the paper  PackDock: a Diffusion Based Side Chain Packing Model for Flexible Protein-Ligand Docking 

If you have any question, feel free to open an issue or reach out to us: [zhangrunze@simm.ac.cn](zhangrunze@simm.ac.cn)✉️.

by [Runze Zhang](https://github.com/Zhang-Runze)
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


# preprocess

[ADFR-Suite](https://ccsb.scripps.edu/adfr/downloads/)

# Retraining DifPocket
Download the data([BC40](https://zenodo.org/) or [PDBbind](https://zenodo.org/)) and place it as described in the "Dataset" section above.
