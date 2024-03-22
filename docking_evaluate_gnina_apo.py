import copy
import os
import glob
import shutil
import torch
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import wandb
from biopandas.pdb import PandasPdb
from rdkit import RDLogger
from torch_geometric.loader import DataLoader,DataListLoader
from torch_geometric.data import Dataset, HeteroData
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from Bio.PDB import PDBParser, Superimposer

from datasets.pdbbind import PDBBind, read_mol, read_pdb
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import gnina_docking, get_model, extract_protein_packing_state, read_first_docking_gnina_log, get_pocket_outside_num, get_ligand_outside_num
from datasets.process_mols import parse_pdb_from_path, read_molecule, extract_receptor_structure, get_rec_graph, get_lig_graph_with_matching
from utils.visualise import PDBFile
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--protein_model_dir', type=str, default='workdir/protein_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--model_dir', type=str, default='workdir/ligand_based_protein_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=20, help='')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--batch_size', type=int, default=3, help='Number of poses to sample in parallel')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
parser.add_argument('--rmsd', action='store_true', default=False, help='Whether to calucate rmsd')

parser.add_argument('--receptor_radius', type=float, default=15, help='Cutoff on distances for receptor edges')
parser.add_argument('--c_alpha_max_neighbors', type=int, default=24, help='Maximum number of neighbors for each residue')
parser.add_argument('--all_atoms', action='store_true', default=True, help='Whether to use the all atoms model')
parser.add_argument('--atom_radius', type=float, default=5, help='Cutoff on distances for atom connections')
parser.add_argument('--atom_max_neighbors', type=int, default=8, help='Maximum number of atom neighbours for receptor')
parser.add_argument('--remove_hs', action='store_true', default=True, help='remove Hs')

parser.add_argument('--matching_popsize', type=int, default=20, help='Differential evolution popsize parameter in matching')
parser.add_argument('--matching_maxiter', type=int, default=20, help='Differential evolution maxiter parameter in matching')
parser.add_argument('--num_conformers', type=int, default=1, help='Number of conformers to match to each ligand')

parser.add_argument('--out_dir', type=str, default="results/test", help='Where to save results to')
parser.add_argument('--seed', type=str, default="666", help='random seed for vina docking')
parser.add_argument('--save_visualisation', action='store_true', default=True, help='Whether to save visualizations')
args = parser.parse_args()




def construct_protein_graph(protein_path):

    protein_graphs = []
    rec_model,pdb_mol = parse_pdb_from_path(protein_path)

    protein_graph = HeteroData()
    protein_graph['name'] = str(PDBID)
    rec, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), lig)

    pocket_outside_num = get_pocket_outside_num(rec_coords)
    pocket_outside_idx = [np.ones_like(arr) if i in pocket_outside_num else arr for i, arr in enumerate(backbone_idx)]

    get_rec_graph(rec, pdb_mol, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, pocket_outside_num, pocket_outside_idx, protein_graph, rec_radius=args.receptor_radius,
                    c_alpha_max_neighbors=args.c_alpha_max_neighbors, all_atoms=args.all_atoms,
                atom_radius=args.atom_radius, atom_max_neighbors=args.atom_max_neighbors, remove_hs=args.remove_hs, lm_embeddings=lm_embeddings)

    
    protein_center = torch.mean(protein_graph['receptor'].pos, dim=0, keepdim=True)
    protein_graph['receptor'].pos -= protein_center
    protein_graph['atom'].pos -= protein_center
    protein_graph.original_center = protein_center
    protein_graphs.append(protein_graph)

    return protein_graphs

def construct_graph(ligand_states_path, loop_num):
    complex_graphs = []
    for idx, ligand_state_path in enumerate(ligand_states_path):
        try:


            packing_num_ = int(ligand_state_path.split("/")[-1].split("_")[3]) +1
            pdbqt_protein_path = f"{out_path}/packing_{PDBID}_{PDBID}_{str(packing_num_)}new.pdbqt"
            pdb_path = f"{out_path}/packing_{PDBID}_{PDBID}_{str(packing_num_)}new.pdb"
            file_extensions = [".pdbqt", ".mol", ".sdf", ".mol2"]
            lig = None

            for ext in file_extensions:
                try:
                    ligand_state_path_ext = ligand_state_path.replace(".pdbqt", ext)
                    if ext == ".pdbqt":
                        lig = read_molecule(ligand_state_path_ext, remove_hs=False, sanitize=False)
                    else:
                        os.system(f"obabel -ipdbqt {ligand_state_path} -O {ligand_state_path_ext}")
                        lig = read_molecule(ligand_state_path_ext, remove_hs=False, sanitize=False)
                        os.remove(ligand_state_path_ext)
                    if lig is not None:
                        break
                except:
                    continue

            
            rec_model, pdb_mol = parse_pdb_from_path(ori_protein_path)

            complex_graph = HeteroData()
            complex_graph['name'] = str(loop_num)+"_"+str(idx+1)
            rec, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), lig)
            cut_off = 5
            ligand_outside_num = get_ligand_outside_num(rec_coords,lig,cut_off)
            ligand_outside_idx = [np.ones_like(arr) if i in ligand_outside_num else arr for i, arr in enumerate(backbone_idx)]

            get_rec_graph(rec, pdb_mol, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, ligand_outside_num, ligand_outside_idx, complex_graph, rec_radius=args.receptor_radius,
                            c_alpha_max_neighbors=args.c_alpha_max_neighbors, all_atoms=args.all_atoms,
                        atom_radius=args.atom_radius, atom_max_neighbors=args.atom_max_neighbors, remove_hs=args.remove_hs, lm_embeddings=lm_embeddings)
            get_lig_graph_with_matching(lig, complex_graph, args.matching_popsize, args.matching_maxiter, False, True,
                                        args.num_conformers, remove_hs=args.remove_hs)

            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            complex_graph['receptor'].pos -= protein_center
            complex_graph['atom'].pos -= protein_center
            complex_graph['ligand'].pos -= protein_center
            complex_graph.original_center = protein_center
            complex_graphs.append(complex_graph)

        except:
            print("error occurred,",pdb_path)
    return complex_graphs, pdb_path

def protein_side_chain_packing(original_data_list, save_visualisation = args.save_visualisation):
    packing_result_path = []
    data_list = []
    for orig_complex_graph in original_data_list:
        for _ in range(6):
            data_list.append(copy.deepcopy(orig_complex_graph))
    randomize_position(data_list, False, False, False)
    confidence_data_list = None
    if save_visualisation:
        visualization_list = []
        for idx, graph in enumerate(data_list):
            mol = Chem.MolFromPDBFile(ori_protein_path, sanitize=True, removeHs=True)
            pdb = PDBFile(mol)
            pdb.add(mol, 0, 0)
            pdb.add((orig_complex_graph['atom'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
            pdb.add((graph['atom'].pos + graph.original_center).detach().cpu(), part=1, order=1)
            visualization_list.append(pdb)
    else :
        visualization_list = None
    data_list, confidence = sampling(data_list=data_list, model=protein_model,
                                    inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                    side_schedule=side_schedule, 
                                    device=protein_device, t_to_sigma=t_to_sigma, model_args=protein_score_model_args,
                                    no_random=args.no_random,
                                    ode=args.ode, visualization_list=visualization_list,
                                    confidence_model=confidence_model,
                                    confidence_data_list=confidence_data_list,
                                    confidence_model_args=confidence_model_args,
                                    batch_size=args.batch_size,
                                    no_final_step_noise=args.no_final_step_noise)
    for idx ,visualization in enumerate(visualization_list):
        visualization.write(f'{out_path}/packing_{data_list[idx]["name"]}_{PDBID}_{idx + 1}.pdb')
        packing_result_path.append(f'{out_path}/packing_{data_list[idx]["name"]}_{PDBID}_{idx + 1}.pdb')
    return packing_result_path

def side_chain_packing(original_data_list, save_visualisation = args.save_visualisation):
    packing_result_path = []
    data_list = []
    for orig_complex_graph in original_data_list:
        for _ in range(6):
            data_list.append(copy.deepcopy(orig_complex_graph))
    randomize_position(data_list, False, False, False)
    confidence_data_list = None
    if save_visualisation:
        visualization_list = []
        for idx, graph in enumerate(data_list):
            mol = Chem.MolFromPDBFile(ori_protein_path, sanitize=True, removeHs=True)
            pdb = PDBFile(mol)
            pdb.add(mol, 0, 0)
            pdb.add((orig_complex_graph['atom'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
            pdb.add((graph['atom'].pos + graph.original_center).detach().cpu(), part=1, order=1)
            visualization_list.append(pdb)
    else :
        visualization_list = None
    data_list, confidence = sampling(data_list=data_list, model=model,
                                    inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                    side_schedule=side_schedule, 
                                    device=complex_device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                    no_random=args.no_random,
                                    ode=args.ode, visualization_list=visualization_list,
                                    confidence_model=confidence_model,
                                    confidence_data_list=confidence_data_list,
                                    confidence_model_args=confidence_model_args,
                                    batch_size=args.batch_size,
                                    no_final_step_noise=args.no_final_step_noise)
    for idx ,visualization in enumerate(visualization_list):
        visualization.write(f'{out_path}/packing_{data_list[idx]["name"]}_{PDBID}_{idx + 1}.pdb')
        packing_result_path.append(f'{out_path}/packing_{data_list[idx]["name"]}_{PDBID}_{idx + 1}.pdb')
    return packing_result_path

def calculate_rmsd(coords1, coords2):
    # Calculate RMSD between two coordinate sets
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    return rmsd

def cluster_proteins(pdb_files, num_clusters):
    # Load PDB structures and calculate coordinates
    parser = PDBParser()
    structures = [parser.get_structure(f"protein_{i}", pdb_file) for i, pdb_file in enumerate(pdb_files)]
    coords = np.array([np.array([atom.get_coord() for atom in structure.get_atoms()]) for structure in structures])

    # Align all structures to the first structure
    superimposer = Superimposer()
    aligned_coords = []
    for i in range(len(coords)):
        fixed_atoms = list(structures[0].get_atoms())
        moving_atoms = list(structures[i].get_atoms())
        superimposer.set_atoms(fixed_atoms, moving_atoms)
        superimposer.apply(structures[i])
        aligned_coords.append(np.array([atom.get_coord() for atom in structures[i].get_atoms()]))

    # Convert aligned coordinates to numpy array
    aligned_coords = np.array(aligned_coords)

    # Calculate pairwise RMSD matrix
    rmsd_matrix = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            rmsd = calculate_rmsd(aligned_coords[i], aligned_coords[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # Perform K-Medoids clustering
    kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=0)
    kmedoids.fit(rmsd_matrix)

    # Get the representative structure for each cluster
    representative_files = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(kmedoids.labels_ == cluster_id)[0]
        representative_index = cluster_indices[np.argmin(np.sum(rmsd_matrix[cluster_indices][:, cluster_indices], axis=1))]
        representative_files.append(pdb_files[representative_index])

    return representative_files

def copy_file(source_file, destination_folder):

    shutil.copy(source_file, destination_folder)
    filename = os.path.basename(source_file)
    new_file_path = os.path.join(destination_folder, filename)
    
    return new_file_path


def extract_rmsd_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        for line in file:
            if line.startswith("RMSD"):
                rmsd_value = line.strip().split()[-1]
                return float(rmsd_value)


def extract_protein_docking_top(docking_dataframe_list, PDBID, out_path):
    ligand_states_path = []
    ligand_mol_states = []
    merged_df = pd.concat(docking_dataframe_list,ignore_index=True)
    sorted_df = merged_df.sort_values('CNNscore')
    for i in range(1,7):
        packing_num = int(sorted_df[i-1:i]['num'])
        ligand_states_path.append(f"{out_path}/{PDBID}_loop_0_{str(packing_num)}_docking_ligand_out_2.pdbqt")
    return ligand_states_path, sorted_df

def extract_protein_docking_top36(docking_dataframe_list, PDBID, out_path):
    ligand_states_path = []
    merged_df = pd.concat(docking_dataframe_list,ignore_index=True)
    sorted_df = merged_df.sort_values('CNNscore')
    for i in range(1,37):
        packing_num = int(sorted_df[i-1:i]['num'])
        vina_mode_num = int(sorted_df[i-1:i]['mode'])
        ligand_states_path.append(f"{out_path}/{PDBID}_loop_0_{str(packing_num)}_docking_ligand_out_{str(vina_mode_num)}.pdbqt")
    return ligand_states_path, sorted_df

"""load score model and its args"""
protein_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
complex_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""load protein score model and ists args"""
with open(f'{args.protein_model_dir}/model_parameters.yml') as f:
    protein_score_model_args = Namespace(**yaml.full_load(f))
protein_t_to_sigma = partial(t_to_sigma_compl, args=protein_score_model_args)
protein_model = get_model(protein_score_model_args, protein_device, t_to_sigma=protein_t_to_sigma, no_parallel=True, only_protein= True)
protein_state_dict = torch.load(f'{args.protein_model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
protein_model.load_state_dict(protein_state_dict, strict=True)
protein_model = protein_model.to(protein_device)
protein_model.eval()

"""load complex score model and ists args"""
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
model = get_model(score_model_args, complex_device, t_to_sigma=t_to_sigma, no_parallel=True)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model = model.to(complex_device)
model.eval()


confidence_model = None
confidence_args = None
confidence_model_args = None
side_schedule = get_t_schedule(inference_steps=args.inference_steps)
os.chdir("./PackDock")
group = "group1"
all_apo_df = pd.read_csv(f"./data/Apo_Holo_fulldata_aligned_pocket_{group}_12a/Apo_Holo_fulldata_aligned_pocket_{group}_12a.txt")

for apo_id in tqdm(all_apo_df['apo_id']):   
    start_time = time.time()
    p_id = apo_id.split("_")[0]
    l_id = apo_id.split("_")[1]
    search_pattern = os.path.join(f"./data/apo2holo_datasets/{group}", p_id, "Ligands", l_id + "*.pdb")
    matching_files = glob.glob(search_pattern)
    SDF_file = matching_files[0]
    mol2_file = matching_files[0]
    ori_LIG_pdb_file = matching_files[0]
    SDF_file = f"./data/Apo_Holo_fulldata_aligned_pocket_{group}_12a_new/{apo_id}/ligand.pdb"
    mol2_file = f"./data/Apo_Holo_fulldata_aligned_pocket_{group}_12a_new/{apo_id}/ligand.pdb"
    ori_LIG_pdb_file = f"./data/Apo_Holo_fulldata_aligned_pocket_{group}_12a_new/{apo_id}/ligand.pdb"
    ori_protein_path = f"./data/Apo_Holo_fulldata_aligned_pocket_{group}_12a_new/{apo_id}/protein_pocket.pdb"

    PDBID = apo_id
    out_path = os.path.join(args.out_dir , PDBID)


    output_dataframe = f"{out_path}/1_dockingscores.csv"

    if not os.path.exists(output_dataframe):
        max_attempts = 5 
        attempt = 1 
        while attempt <= max_attempts:

            try:
                lig = Chem.MolFromPDBFile(ori_LIG_pdb_file, sanitize=True, removeHs=True) 
                if not os.path.exists(out_path):
                    os.system("mkdir "+out_path)
                protein_path = f"./data/Apo_Holo_fulldata_aligned_pocket_{group}_12a/{apo_id}/protein_pocket.pdb"

                """Conformational selection stage"""
                protein_graphs = construct_protein_graph(ori_protein_path)
                packing_result_path = protein_side_chain_packing(protein_graphs)
                protein_docking_dataframes = []
                os.chdir(os.path.dirname(__file__))
                loop_num = 0 
                for idx, packing_protein_path in enumerate(packing_result_path):
                    protein_path = extract_protein_packing_state(packing_protein_path)
                    ligand_states_path ,protein_path = gnina_docking(protein_path, SDF_file, out_path, loop_num, PDBID, args.seed, idx)
                    os.chdir(os.path.dirname(__file__))
                    protein_log_path = f"{out_path}/{PDBID}_0_{idx}_docking_ligand_gnina_score_out.log"
                    if args.rmsd:
                        os.chdir(os.path.dirname(__file__))
                        protein_docking_df = read_first_docking_gnina_log(protein_log_path)
                        os.chdir(os.path.dirname(__file__))
                        protein_docking_rmsd = []
                        nums = []
                        for ligand_state_path in ligand_states_path:
                            rmsd_txt = f"{out_path}/rmsd.txt"
                            os.system(f'obrms {SDF_file} {ligand_state_path } > {rmsd_txt}')
                            rmsd = extract_rmsd_from_txt(rmsd_txt)
                            protein_docking_rmsd.append(rmsd)
                            nums.append(idx)
                        protein_docking_df = protein_docking_df.head(len(protein_docking_rmsd))
                        protein_docking_df['true_rmsd'] = protein_docking_rmsd
                        protein_docking_df['num'] = nums
                        # protein_docking_df.to_csv(f"{out_path}/0_{idx}_dockingscores.csv", index=False)
                        protein_docking_dataframes.append(protein_docking_df)
                ligand_states_path, sorted_df = extract_protein_docking_top(protein_docking_dataframes, PDBID, out_path)
                sorted_df.to_csv(f"{out_path}/0_dockingscores.csv", index=False)
                os.chdir(os.path.dirname(__file__))
                protein_docking_dataframes = []
                loop_num +=1

                """induced fit stage"""
                complex_graphs, pdb_path = construct_graph(ligand_states_path, loop_num)
                score_list = []
                k_means_file_list = []
                packing_result_path = side_chain_packing(complex_graphs)
                for idx, packing_protein_path in enumerate(packing_result_path):
                    os.chdir(os.path.dirname(__file__))
                    protein_path = extract_protein_packing_state(packing_protein_path)
                    ligand_states_path ,pdbqt_protein_path = gnina_docking(protein_path, SDF_file, out_path, loop_num, PDBID, args.seed, idx)
                    os.chdir(os.path.dirname(__file__))
                    protein_log_path = f"{out_path}/{PDBID}_1_{idx}_docking_ligand_gnina_score_out.log"
                    if args.rmsd:
                        protein_docking_df = read_first_docking_gnina_log(protein_log_path)
                        os.chdir(os.path.dirname(__file__))
                        protein_docking_rmsd = []
                        nums = []
                        for ligand_state_path in ligand_states_path:
                            rmsd_txt = f"{out_path}/rmsd.txt"
                            os.system(f'obrms {SDF_file} {ligand_state_path } > {rmsd_txt}')
                            rmsd = extract_rmsd_from_txt(rmsd_txt)
                            protein_docking_rmsd.append(rmsd)
                            nums.append(idx)
                        protein_docking_df = protein_docking_df.head(len(protein_docking_rmsd))
                        protein_docking_df['true_rmsd'] = protein_docking_rmsd
                        protein_docking_df['num'] = nums

                        protein_docking_dataframes.append(protein_docking_df)
                merged_df = pd.concat(protein_docking_dataframes,ignore_index=True)
                sorted_df = merged_df.sort_values('true_rmsd')
                sorted_df.to_csv(f"{out_path}/1_dockingscores.csv", index=False)

                break
            except Exception as e:
                    print(f"{apo_id} error occurred, attempt {attempt}." )
                    attempt += 1