import os
import subprocess
import warnings
from datetime import datetime
import signal
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import RemoveHs, MolToPDBFile ,AllChem
from rdkit.Geometry import Point3D
from torch_geometric.nn.data_parallel import DataParallel

from models.all_atom_score_model import TensorProductScoreModel as AAScoreModel
from models.all_atom_score_model import TensorProduct_protein_ScoreModel
from utils.diffusion_utils import get_timestep_embedding
from spyrmsd import rmsd, molecule

from openbabel import openbabel as ob
from openbabel import pybel
from Bio.PDB import *

def kabsch_rmsd(P, Q, N, rotated_atom_num):
    """
    Use the Kabsch algorithm to align Q to P and calculate their RMSD

    Parameters:
    P: ndarray, a two-dimensional array of shape (N, 3), representing the reference coordinates
    Q: ndarray, a two-dimensional array of shape (N, 3), representing the coordinates to be aligned

    Return value:
    rmsd: float, the aligned RMSD value
    """
    rmsd_list= []
    for i in range(N):
        _P = P[0,:,:]
        _Q = Q[i,:,:]
        centroid_P = np.mean(_P, axis=0)
        centroid_Q = np.mean(_Q, axis=0)
        _P -= centroid_P
        _Q -= centroid_Q
        cov = np.dot(_Q.T, _P)
        U, S, Vt = np.linalg.svd(cov)
        R = np.dot(Vt.T, U.T)
        _Q = np.dot(_Q, R.T)
        _Q += centroid_P
        _ = np.mean((_P - _Q)**2, axis= 1)
        rmsd = np.sqrt(_.sum()/rotated_atom_num)
        rmsd = rmsd.squeeze()
        rmsd_list.append(np.array(rmsd))
    rmsd_all = np.asarray(rmsd_list)
    return rmsd_all


def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f') if cache_name is None else cache_name
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, '.openbabel_cache/obrmsd_mol1_cache.pdb')
        mol1_path = '.openbabel_cache/obrmsd_mol1_cache.pdb'
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, '.openbabel_cache/obrmsd_mol2_cache.pdb')
        mol2_path = '.openbabel_cache/obrmsd_mol2_cache.pdb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd",
                                     shell=True)
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float)


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_optimizer_and_scheduler(args, model, scheduler_mode='min'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False, only_protein=False):
    if 'all_atoms' in args and args.all_atoms:
        model_class = AAScoreModel

    if only_protein:
        print("Conformational Selection Stage")
        model_class = TensorProduct_protein_ScoreModel
    else:
        print("Induced-fit Stage")

    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale)

    lm_embedding_type = None
    if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

    model = model_class(t_to_sigma=t_to_sigma,
                        device=device,
                        no_torsion=args.no_torsion,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=args.num_conv_layers,
                        lig_max_radius=args.max_radius,
                        scale_by_sigma=args.scale_by_sigma,
                        sigma_embed_dim=args.sigma_embed_dim,
                        ns=args.ns, nv=args.nv,
                        distance_embed_dim=args.distance_embed_dim,
                        cross_distance_embed_dim=args.cross_distance_embed_dim,
                        batch_norm=not args.no_batch_norm,
                        dropout=args.dropout,
                        use_second_order_repr=args.use_second_order_repr,
                        cross_max_distance=args.cross_max_distance,
                        dynamic_max_cross=args.dynamic_max_cross,
                        lm_embedding_type=lm_embedding_type,
                        confidence_mode=confidence_mode,
                        num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1)

    if device.type == 'cuda' and not no_parallel:
        model = DataParallel(model)
    model.to(device)
    return model


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    mol = molecule.Molecule.from_rdkit(mol)
    mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
    mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
    mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
    RMSD = rmsd.symmrmsd(
        coords1,
        coords2,
        mol.atomicnums,
        mol2_atomicnums,
        mol.adjacency_matrix,
        mol2_adjacency_matrix,
    )
    return RMSD


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]



def sdf2smi (SDF_file):
    conv = ob.OBConversion()
    mol = ob.OBMol()
    conv.SetInAndOutFormats("sdf", "smiles")
    conv.ReadFile(mol, SDF_file)
    smiles = conv.WriteString(mol)
    return smiles

def smi2mol2(smiles, out_path):
    if not os.path.exists(out_path):
        os.system("mkdir "+out_path)
    PDBID = out_path.split("/")[0]
    mol = pybel.readstring('smi', smiles)
    mol.make3D()
    mol.localopt(forcefield='uff')
    out_file = os.path.join(out_path, f"{PDBID}_ligand.mol2")   #test_outdir/1a0q/1a0q_ligand.mol2
    mol.write('mol2', out_file, overwrite=True)
    return out_file

def pdbqt2mol2(pdbqt_path, remove_source_file = False):

    obConversion = ob.OBConversion()
    obConversion.SetInFormat("pdbqt")
    obConversion.SetOutFormat("mol2")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, pdbqt_path)
    new_mol2_path = pdbqt_path.replace('.pdbqt', '.mol2')
    obConversion.WriteFile(mol, new_mol2_path)
    if remove_source_file == True:
        os.remove(pdbqt_path)

    return new_mol2_path

def pdbqt2pdb(pdbqt_path, remove_source_file = False):
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("pdbqt")
    obConversion.SetOutFormat("pdb")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, pdbqt_path)
    new_pdb_path = pdbqt_path.replace('.pdbqt', '.pdb')
    obConversion.WriteFile(mol, new_pdb_path)
    if remove_source_file == True:
        os.remove(pdbqt_path)

    return new_pdb_path

def get_first_docking_config(out_protein_path, out_ligand_path, out_path, pdb_name, seed, pocket_size = 30, mode = "32"):
    ori_coords = get_coor_from_pdbqt(out_protein_path)
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])

    new_config = os.path.join(out_path,pdb_name+"_first_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size[0],size[1],size[2]))
        if mode in ['detail', 'balance']:
            f.write("search_mode = {}\n".format(mode))
        else:
            f.write("exhaustiveness = {}\n".format(mode.split('_')[0]))
        
        f.write("receptor = {}\nligand = {}\n".format(out_protein_path, out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 6\nverbosity = 1\n".format(os.path.join(out_path,pdb_name+"_first_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join(out_path,pdb_name+"_first_docking_ligand_out.pdbqt")
    return new_config, vina_out

def get_loop_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    ori_coords = get_coor_from_pdbqt(protein_path)
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])

    new_config = os.path.join(out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size[0],size[1],size[2]))
        if mode in ['detail', 'balance']:
            f.write("search_mode = {}\n".format(mode))
        else:
            f.write("exhaustiveness = {}\n".format(mode.split('_')[0]))
        f.write("receptor = {}\nligand = {}\n".format(protein_path, out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 6\nverbosity = 1\n".format(os.path.join(out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join(out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out

def get_new_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    ori_coords = get_coor_from_pdbqt(protein_path)
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])

    new_config = os.path.join(out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size[0],size[1],size[2]))
        if mode in ['detail', 'balance']:
            f.write("search_mode = {}\n".format(mode))
        else:
            f.write("exhaustiveness = {}\n".format(mode.split('_')[0]))
        f.write("receptor = {}\nligand = {}\n".format(protein_path, out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 1\nverbosity = 1\n".format(os.path.join(out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join(out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out

def get_gnina_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    vina_out = os.path.join(out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    new_config = os.path.join(out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        
        f.write("autobox_ligand = {}\nautobox_add = 4\n".format(out_ligand_path))
        if mode in ['detail', 'balance']:
            f.write("search_mode = {}\n".format(mode))
        else:
            f.write("exhaustiveness = {}\n".format(mode.split('_')[0]))
        f.write("receptor = {}\nligand = {}\n".format(protein_path, out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 1\n".format(vina_out, seed))
        
    return new_config, vina_out


def get_gpu_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    ori_coords = get_coor_from_pdbqt(protein_path)
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])


    new_config = os.path.join("./"+out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size[0],size[1],size[2]))
        f.write("thread = 8000\n")
        f.write("receptor = {}\nligand = {}\n".format("./"+protein_path, "./"+out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 1\n".format(os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out

def get_gpu_docking_baseline_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    ori_coords = get_coor_from_pdbqt(protein_path)
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])-5

    new_config = os.path.join("./"+out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(min(25,max(size[0],15)),min(25,max(size[1],15)),min(25,max(size[2],15))))
        f.write("thread = 8000\n")
        f.write("receptor = {}\nligand = {}\n".format("./"+protein_path, "./"+out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 36\n".format(os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out

def get_gpu_docking_config_with_ligand_boxsize(protein_path, out_ligand_path, ori_LIG_pdb_file, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    ori_coords = get_coordinates_from_pdb(ori_LIG_pdb_file)
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])+4

    new_config = os.path.join("./"+out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(max(size[0],10),max(size[1],10),max(size[2],10)))
        f.write("thread = 8000\n")
        f.write("receptor = {}\nligand = {}\n".format("./"+protein_path, "./"+out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 1\n".format(os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out


def get_gpu_docking_config_with_ligand_boxsize_sdf(protein_path, out_ligand_path, sdf_file, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    _, ligand_extension = os.path.splitext(sdf_file)
    if sdf_file.endswith('.sdf'):
        mol_supplier = Chem.SDMolSupplier(sdf_file,sanitize = False)
        mol = mol_supplier[0]
    if sdf_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(sdf_file,sanitize = False) 
    if sdf_file.endswith('.mol'):
        mol = Chem.MolFromMolFile(sdf_file,sanitize = False) 
    if sdf_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(sdf_file,sanitize = False) 

    min_x, min_y, min_z = 1e4, 1e4, 1e4
    max_x, max_y, max_z = -1e4, -1e4, -1e4
    if mol is not None:
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            x, y, z = pos.x, pos.y, pos.z
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2
    size_x = max_x - min_x + 2
    size_y = max_y - min_y + 2
    size_z = max_z - min_z + 2



    new_config = os.path.join("./"+out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center_x,center_y,center_z))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size_x,size_y,size_z))
        f.write("thread = 8000\n")
        f.write("receptor = {}\nligand = {}\n".format("./"+protein_path, "./"+out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 1\n".format(os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out


def get_gpu_docking_flex_config_with_ligand_boxsize_sdf(rigid_protein_file, flex_protein_file, out_ligand_path, sdf_file, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    mol_supplier = Chem.SDMolSupplier(sdf_file,sanitize = False)
    min_x, min_y, min_z = 1e4, 1e4, 1e4
    max_x, max_y, max_z = -1e4, -1e4, -1e4
    for mol in mol_supplier:
        if mol is not None:
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                x, y, z = pos.x, pos.y, pos.z
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                min_z = min(min_z, z)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                max_z = max(max_z, z)
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2
    size_x = max_x - min_x + 10
    size_y = max_y - min_y + 10
    size_z = max_z - min_z + 10



    new_config = os.path.join("./"+out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center_x,center_y,center_z))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size_x,size_y,size_z))
        f.write("thread = 8000\n")
        f.write("receptor = {}\nflex = {}\nligand = {}\n".format("./"+rigid_protein_file, "./"+flex_protein_file, "./"+out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 36\n".format(os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out


def get_gpu_docking_baseline_config_with_ligand_boxsize(protein_path, out_ligand_path, ori_LIG_pdb_file, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    ori_coords = get_coordinates_from_pdb(ori_LIG_pdb_file)
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])+4

    new_config = os.path.join("./"+out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size[0],size[1],size[2]))
        f.write("thread = 8000\n")
        f.write("receptor = {}\nligand = {}\n".format("./"+protein_path, "./"+out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 36\n".format(os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out

def get_gpu_docking_config_with_smaller_boxsize(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed, pocket_size = 30, mode = "32"):
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    ori_coords = get_coor_from_pdbqt(protein_path)
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])-10

    new_config = os.path.join("./"+out_path,pdb_name+"_loop_docking_config.txt")
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(min(size[0],20),min(size[1],20),min(size[2],20)))
        f.write("thread = 8000\n")
        f.write("receptor = {}\nligand = {}\n".format("./"+protein_path, "./"+out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 1\n".format(os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt"), seed))
        vina_out = os.path.join("./"+out_path,pdb_name+f"_loop_{loop_num}_{idx}_docking_ligand_out.pdbqt")
    return new_config, vina_out


def pdb2pdbqt(protein_path,ligand_path, remove_ligand=False, remove_protein=False):

    if protein_path is not None:
        if protein_path.endswith('.pdbqt') is not True:
            protein_extension = os.path.splitext(protein_path)[1]  
            ADFR_protein = "./ADFR_suite/bin/prepare_receptor"
            out_protein_path = protein_path.replace(protein_extension, ".pdbqt")
            os.system(f"{ADFR_protein} -r {protein_path} -o {out_protein_path} -e True -A 'hydrogens'")
            if remove_protein:
                os.remove(protein_path)
            protein_path = out_protein_path
    if ligand_path is not None:
        if ligand_path.endswith('.pdbqt') is not True:
            ligand_extension = os.path.splitext(ligand_path)[1]  
            ADFR_ligand = "./ADFR_suite/bin/prepare_ligand"
            remove_path = os.path.abspath(os.getcwd()) #like ./PackDock
            os.system(f"mv {ligand_path} {remove_path}")
            new_ligand_path = os.path.join(remove_path,ligand_path.split("/")[-1])
            out_ligand_path = ligand_path.replace(ligand_extension, ".pdbqt")
            os.system(f"{ADFR_ligand} -l {new_ligand_path} -o {out_ligand_path} -A 'hydrogens' ")
            if remove_ligand:
                os.remove(new_ligand_path)
            ligand_path = out_ligand_path


    return protein_path, ligand_path

def first_docking(protein_path, ligand_path, out_path, pdb_name, seed):
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        new_config, vina_out = get_first_docking_config(out_protein_path, out_ligand_path, out_path, pdb_name, seed)
        vina_path = "your vina path" #like ./vina/vina
        os.system(f"{vina_path} --config {new_config}  > {out_path}/{pdb_name}_first_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path

def loop_docking(protein_path_list, ligand_path, out_path, loop_num, pdb_name, seed):
    with time_limit(300):
        _, out_ligand_path = pdb2pdbqt(None, ligand_path, remove_ligand=True)
        log_paths = []
        vina_out_path = []
        for idx, protein_path in enumerate(protein_path_list):
            new_config, vina_out = get_loop_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed)
            vina_out_path.append(vina_out)
            vina_path = "your vina path" #like ./vina/vina
            os.system(f"{vina_path} --config {new_config}  > {out_path}/{pdb_name}_loop_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
            log_paths.append(f"{out_path}/{pdb_name}_loop_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        rank_num_list, vina_score_max = log2rank(log_paths, out_path, loop_num)
        states_path = []
        for i in rank_num_list:
            (quotient, remainder) = divmod(int(i),6)
            states_path.append(split_pdbqt_file_inloop(vina_out_path[quotient], remainder+1))
        for path in vina_out_path:
            os.remove(path)

        return states_path, vina_score_max
    
def protein_docking(protein_path, ligand_path, out_path, pdb_name, seed, idx):
    loop_num = str(0)
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        new_config, vina_out = get_loop_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed)
        vina_path = "your vina path" #like ./vina/vina
        os.system(f"{vina_path} --config {new_config}  > {out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path

def new_docking(protein_path, ligand_path, out_path, loop_num, pdb_name, seed, idx):
    # loop_num = str(0)
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        new_config, vina_out = get_new_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed)
        vina_path = "your vina path" #like ./vina/vina
        os.system(f"{vina_path} --config {new_config}  > {out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path
    
def gnina_docking(protein_path, ligand_path, out_path, loop_num, pdb_name, seed, idx):
    with time_limit(300):
        new_config, docking_ligand = get_gnina_docking_config(protein_path, ligand_path, out_path, pdb_name, idx, loop_num, seed)
        gnina_path = "your gnina path" #like ./gnina
        os.system(f"{gnina_path} --config {new_config}  > {out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_gnina_score_out.log")
        states_path = split_pdbqt_file(docking_ligand)
        return states_path, protein_path
        

def gpu_docking(protein_path, ligand_path, out_path, loop_num, pdb_name, seed, idx):
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        vina_path = "your vina-GPU path" #like ./Vina-GPU-2.0/Vina-GPU+/Vina-GPU"
        new_config, vina_out = get_gpu_docking_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed)
        os.chdir("your vina-GPU dir path") #like ./Vina-GPU-2.0/Vina-GPU+
        os.system(f"{vina_path} --config {new_config}  > ./{out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path
    
def gpu_docking_baseline(protein_path, ligand_path, out_path, loop_num, pdb_name, seed, idx):
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        vina_path = "your vina-GPU path" #like ./Vina-GPU-2.0/Vina-GPU+/Vina-GPU"
        new_config, vina_out = get_gpu_docking_baseline_config(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed)
        os.chdir("your vina-GPU dir path") #like ./Vina-GPU-2.0/Vina-GPU+
        os.system(f"{vina_path} --config {new_config}  > ./{out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path

def gpu_docking_baseline_with_ligand_boxsize(protein_path, ligand_path, ori_LIG_pdb_file, out_path, loop_num, pdb_name, seed, idx):
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        vina_path = "your vina-GPU path" #like ./Vina-GPU-2.0/Vina-GPU+/Vina-GPU"
        new_config, vina_out = get_gpu_docking_baseline_config_with_ligand_boxsize(protein_path, out_ligand_path, ori_LIG_pdb_file, out_path, pdb_name, idx, loop_num, seed)
        os.chdir("your vina-GPU dir path") #like ./Vina-GPU-2.0/Vina-GPU+
        os.system(f"{vina_path} --config {new_config}  > ./{out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path
    
def gpu_docking_with_ligand_boxsize(protein_path, ligand_path, ori_LIG_pdb_file, out_path, loop_num, pdb_name, seed, idx):
    # loop_num = str(0)
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        
        vina_path = "your vina-GPU path" #like ./Vina-GPU-2.0/Vina-GPU+/Vina-GPU"
        new_config, vina_out = get_gpu_docking_config_with_ligand_boxsize(protein_path, out_ligand_path, ori_LIG_pdb_file, out_path, pdb_name, idx, loop_num, seed)
        os.chdir("your vina-GPU dir path") #like ./Vina-GPU-2.0/Vina-GPU+
        os.system(f"{vina_path} --config {new_config}  > ./{out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path
    
def gpu_docking_with_ligand_boxsize_sdf(protein_path, ligand_path, ori_LIG_sdf_file, out_path, loop_num, pdb_name, seed, idx):
    # loop_num = str(0)
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        vina_path = "your vina-GPU path" #like ./Vina-GPU-2.0/Vina-GPU+/Vina-GPU"
        new_config, vina_out = get_gpu_docking_config_with_ligand_boxsize_sdf(protein_path, out_ligand_path, ori_LIG_sdf_file, out_path, pdb_name, idx, loop_num, seed)
        os.chdir("your vina-GPU dir path") #like ./Vina-GPU-2.0/Vina-GPU+
        os.system(f"{vina_path} --config {new_config}  > ./{out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path

def gpu_docking_flex_with_ligand_boxsize_sdf(rigid_protein_file, flex_protein_file, ligand_path, ori_LIG_sdf_file, out_path, loop_num, pdb_name, seed, idx):
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(rigid_protein_file, ligand_path, remove_ligand=True)
        vina_path = "your vina-GPU path" #like ./Vina-GPU-2.0/Vina-GPU+/Vina-GPU"
        new_config, vina_out = get_gpu_docking_flex_config_with_ligand_boxsize_sdf(rigid_protein_file, flex_protein_file, out_ligand_path, ori_LIG_sdf_file, out_path, pdb_name, idx, loop_num, seed)
        os.chdir("your vina-GPU dir path") #like ./Vina-GPU-2.0/Vina-GPU+
        os.system(f"{vina_path} --config {new_config}  > ./{out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path

def gpu_docking_with_smaller_boxsize(protein_path, ligand_path, out_path, loop_num, pdb_name, seed, idx):
    # loop_num = str(0)
    with time_limit(300):
        out_protein_path, out_ligand_path = pdb2pdbqt(protein_path, ligand_path, remove_ligand=True)
        vina_path = "your vina-GPU path" #like ./Vina-GPU-2.0/Vina-GPU+/Vina-GPU"
        new_config, vina_out = get_gpu_docking_config_with_smaller_boxsize(protein_path, out_ligand_path, out_path, pdb_name, idx, loop_num, seed)
        os.chdir("your vina-GPU dir path") #like ./Vina-GPU-2.0/Vina-GPU+
        os.system(f"{vina_path} --config {new_config}  > ./{out_path}/{pdb_name}_{loop_num}_{idx}_docking_ligand_vina_score_out.log")
        states_path = split_pdbqt_file(vina_out)
        os.remove(vina_out)
        return states_path, out_protein_path

def log2rank(log_paths, out_path, loop_num):
    num = 0
    score_list = []
    for log_path in log_paths:
        with open(log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("   1") or line.startswith("   2") or line.startswith("   3") or line.startswith("   4") or line.startswith("   5") or line.startswith("   6"):
                    score_list.append((num, float(line.split()[1])))
                    num += 1
        os.remove(log_path)
    docking_score_df = pd.DataFrame(score_list, columns=['Num', 'Score'])
    docking_score_df = docking_score_df.sort_values('Score')
    docking_score_df.to_csv(f'{out_path}/{loop_num}_dockingscores.csv', index=False)
    rank_num_list = []
    for i in docking_score_df['Num'][:6]:
        rank_num_list.append(i)
    vina_score_max = docking_score_df['Score'][:1]
    return rank_num_list, vina_score_max

def get_coor_from_protein_pdbqt(ligfile: str) -> dict:
    '''
    Get coordinates of the atoms in the ligand from a pdb file.
    
    Args:
      ligfile (str): the pdbqt file of the ligand
    
    Returns:
      A list of coordinates of atoms in the ligand.
    '''
    with open(ligfile, "r") as f:
        lines = f.readlines()
    coor = []
    for line in lines:
        if len(line) > 60 and line[0:4]=='ATOM':
            # print(line)
            if line[13] == 'H':
                continue
            coor.append([
                float(line[30:38].strip()), 
                float(line[38:46].strip()), 
                float(line[46:54].strip())
            ])
        if line[:7] == 'MODEL 2':
            break
    # print(coor)
    return coor


def get_coordinates_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    coordinates = []
    # 获取原子坐标信息
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords = atom.get_coord()
                    coordinates.append(coords)
    
    return coordinates



def get_coor_from_pdbqt(ligfile: str) -> dict:
    '''
    Get coordinates of the atoms in the ligand from a pdb file.
    
    Args:
      ligfile (str): the pdbqt file of the ligand
    
    Returns:
      A list of coordinates of atoms in the ligand.
    '''
    with open(ligfile, "r") as f:
        lines = f.readlines()
    coor = []
    for line in lines:
        if len(line) > 60 and line[0:4]=='ATOM':
            if line[13] == 'H':
                continue
            coor.append([
                float(line[30:38].strip()), 
                float(line[38:46].strip()), 
                float(line[46:54].strip())
            ])
        if line[:7] == 'MODEL 2':
            break
    return coor


def split_pdbqt_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    current_state = None
    state_lines = []
    states_path = []
    for line in lines:
        if line.startswith("MODEL"):
            if current_state is not None:
                state_path = f"{file_path[:-6]}_{current_state}.pdbqt"
                states_path.append(state_path)
                with open(state_path, "w") as f_out:
                    f_out.write("".join(state_lines))
                state_lines = []
            current_state = line.strip().split()[1]
        elif line.startswith("ENDMDL"):
            continue
        else:
            state_lines.append(line)

    if current_state is not None:
        state_path = f"{file_path[:-6]}_{current_state}.pdbqt"
        states_path.append(state_path)
        with open(state_path, "w") as f_out:
            f_out.write("".join(state_lines))
    return states_path

def split_pdb_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    current_state = None
    state_lines = []
    states_path = []
    for line in lines:
        if line.startswith("MODEL"):
            if current_state is not None:
                state_path = file_path   #f"{file_path[:-4]}.pdb"
                states_path.append(state_path)
                with open(state_path, "w") as f_out:
                    f_out.write("".join(state_lines))
                state_lines = []
            current_state = line.strip().split()[1]
        elif line.startswith("ENDMDL"):
            continue
        else:
            state_lines.append(line)

    if current_state is not None:
        state_path = f"{file_path[:-4]}.pdb"
        states_path.append(state_path)
        with open(state_path, "w") as f_out:
            f_out.write("".join(state_lines))
    return states_path

def split_pdbqt_file_inloop(file_path, model_num):

    output_file = f"{file_path[:-6]}_{model_num}.pdbqt"
    with open(file_path) as f_in, open(output_file, 'w') as f_out:
        current_model = None
        for line in f_in:
            if line.startswith('MODEL'):
                current_model = int(line.split()[1])
                if current_model != model_num:
                    continue
            elif line.startswith('ENDMDL'):
                current_model = None
            elif current_model == model_num:
                f_out.write(line)
    return output_file

def read_pdbqt(filename):
    with open(filename, "r") as f:
        pdbqt_string = f.read()
    mol = AllChem.Docking.readDockingPose(pdbqt_string)
    return mol


def cal_vina_score(protein_path, ligand_path, out_path, pdb_name, seed):
    original_name = os.path.splitext(protein_path)[0]
    new_config = get_config(protein_path, ligand_path, out_path, pdb_name, original_name, seed)
    vina_path = "your vina path" #like ./vina/vina
    os.system(f"{vina_path} --config {new_config} --score_only > {original_name}_vina_score_out.log")
    os.remove(new_config)
    log_path = f"{original_name}_vina_score_out.log"
    score = vinalog2score(log_path)

    return (original_name, float(score))

def vinalog2score(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("Estimated Free Energy"):
            score = line.split(" ")[-3]
            os.remove(log_path)
    return score

def extract_protein_packing_state(protein_path):
    with open(protein_path, 'r') as f:
        protein_lines = f.readlines()
    protein_states = {}
    state_num = 0
    for line in protein_lines:
        if line.startswith('MODEL'):
            state_num += 1
            protein_states[0] = []
        if line.startswith('ATOM'): 
            protein_states[0].append(line)
    new_protein_path = protein_path.replace(".pdb", "new.pdb")
    with open(new_protein_path, 'w') as f:
        for state_num, protein_state in protein_states.items():
            f.write(f'MODEL {state_num}\n')
            f.writelines(protein_state)
            f.write('ENDMDL\n')
    os.remove(protein_path)
    return new_protein_path

def extract_ligand_docking_state(ligand_sdf_path, loop_num):
    folder_path = os.path.dirname(ligand_sdf_path)
    new_ligand_path = []
    PDBID = ligand_sdf_path.split("/")[-1].split("_")[0]
    with open(ligand_sdf_path, 'r') as f:
        lines = f.readlines()
    conformer_count = 0
    for i, line in enumerate(lines):
        if PDBID in line :
            conformer_count += 1
            out_filename = f'{folder_path}/loop_{str(loop_num)}_conformer_{conformer_count}.sdf'
            out_file = open(out_filename, 'w')
        out_file.write(line)
        if '$$$$' in line:
            out_file.close()
            new_ligand_path.append(out_filename)
    return new_ligand_path

def extract_ligand_docking_state_inloop(ligand_sdf_path, loop_num, packing_num):
    folder_path = os.path.dirname(ligand_sdf_path)
    new_ligand_path = []
    PDBID = ligand_sdf_path.split("/")[-1].split("_")[0]
    with open(ligand_sdf_path, 'r') as f:
        lines = f.readlines()
    conformer_count = 0
    for i, line in enumerate(lines):
        if PDBID in line :
            conformer_count += 1
            out_filename = f'{folder_path}/loop_{str(loop_num)}_packing_{packing_num}_conformer_{conformer_count}.sdf'
            out_file = open(out_filename, 'w')
        out_file.write(line)
        if '$$$$' in line:
            out_file.close()
            new_ligand_path.append(out_filename)

    return new_ligand_path

def get_config(out_protein_path, out_ligand_path, out_path, pdb_name, original_name, seed, normal_pocket = False, pocket_size = 30, mode = "32"):
    ori_coords = get_coor_from_pdbqt(out_protein_path)
    center_max = [-1e4,-1e4,-1e4]
    center_min = [1e4,1e4,1e4]
    center = [0,0,0]
    size = [0,0,0]
    for co in ori_coords:
        for i in range(3):
            center_max[i] = max(center_max[i], co[i])
            center_min[i] = min(center_min[i], co[i])
    for i in range(3):
        center[i] = (center_max[i] + center_min[i]) / 2
        size[i] = (center_max[i] - center_min[i])
    new_config = f"{original_name}_config.txt"
    with open(new_config,"w") as f:
        f.write("center_x = {}\ncenter_y = {}\ncenter_z = {}\n".format(center[0],center[1],center[2]))
        f.write("size_x = {}\nsize_y = {}\nsize_z = {}\n".format(size[0],size[1],size[2]))
        if mode in ['detail', 'balance']:
            f.write("search_mode = {}\n".format(mode))
        else:
            f.write("exhaustiveness = {}\n".format(mode.split('_')[0]))
        f.write("receptor = {}\nligand = {}\n".format(out_protein_path, out_ligand_path))
        f.write("out = {}\nseed = {}\nnum_modes = 9\nverbosity = 1\n".format(f"{original_name}_ligand_out.pdbqt", seed))
    return new_config


def get_mol_coords_from_sdf(sdf_path):
    mol_sdf = Chem.SDMolSupplier(sdf_path)[0]
    if mol_sdf == None:
        mol2_path = sdf_path.replace(".sdf", ".mol2")
        mol_sdf = Chem.MolFromMol2File(mol2_path)
        if mol_sdf == None:
            mol_path = sdf_path.replace(".sdf", ".mol")
            mol_sdf = Chem.MolFromMolFile(mol_path,sanitize= False)
            if mol_sdf == None:
                lig_pdb = mol2_path.replace("_ligand.mol2", "_LIG.pdb")
                mol_sdf = Chem.MolFromPDBFile(lig_pdb, sanitize=True, removeHs=True) 
    mol_sdf = Chem.RemoveHs(mol_sdf)
    sdf_conf = mol_sdf.GetConformer()
    sdf_coords = sdf_conf.GetPositions()
    sdf_coords = np.array(sdf_coords).reshape(-1, 3)
    return mol_sdf, sdf_coords

def get_mol_coords_from_pdb(pdb_path):
    mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=True)
    mol = Chem.RemoveHs(mol)
    mol_conf = mol.GetConformer()
    coords = mol_conf.GetPositions()
    coords = np.array(coords).reshape(-1, 3)
    return mol, coords

def get_mol_coords_from_mol2(path):
    file_extensions = [".pdb", ".mol2", ".mol", ".sdf", ".pdbqt"]
    mol = None
    for ext in file_extensions:
        try:
            if ext ==".pdb":
                pdb_path = path.replace(".mol2", ".pdb")
                mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=True)
            elif ext == ".mol2":
                mol = Chem.MolFromMol2File(path)
            elif ext == ".mol":
                mol_path = path.replace(".mol2", ".mol")
                mol = Chem.MolFromMolFile(mol_path)
            elif ext == ".sdf":
                sdf_path = path.replace(".mol2", ".sdf")
                mol = Chem.SDMolSupplier(sdf_path)[0]
            elif ext == ".pdbqt":
                pdbqt_path = path.replace(".mol2", ".pdbqt")
                mol = Chem.MolFromPDBFile(pdbqt_path, sanitize=True, removeHs=True) 
            if mol is not None:
                mol = Chem.RemoveHs(mol)
                break
        except:
            continue

    conf = mol.GetConformer()
    coords = conf.GetPositions()
    coords = np.array(coords).reshape(-1, 3)
    return mol, coords

def get_mol_coords_from_pdbqt(pdbqt_path):
    with open(pdbqt_path, 'r') as f:
        pdbqt_block = f.read()
    mol_pdbqt = Chem.MolFromPDBBlock(pdbqt_block, sanitize=False)
    pdbqt_conf = mol_pdbqt.GetConformer()
    pdbqt_coords = pdbqt_conf.GetPositions()
    pdbqt_coords = np.array(pdbqt_coords).reshape(-1, 3)
    return mol_pdbqt, pdbqt_coords

def read_first_docking_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()[10:]
    df = pd.DataFrame(columns=['mode', 'affinity', 'rmsd_l', 'rmsd_u'])
    for line in lines:
        if line.startswith('   1') or line.startswith('   2') or line.startswith('   3') or line.startswith('   4') or line.startswith('   5') or line.startswith('   6'):
            data = line.split()
            df = df.append({'mode': int(data[0]), 
                            'affinity': float(data[1]), 
                            'rmsd_l': float(data[2]), 
                            'rmsd_u': float(data[3])}, 
                        ignore_index=True)
    return df

def read_first_docking_gnina_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()[10:]
    df = pd.DataFrame(columns=['mode', 'minimizedAffinity', 'CNNscore', 'CNNaffinity'])
    for line in lines:
        if line.startswith('    1') or line.startswith('    2') or line.startswith('    3') or line.startswith('    4') or line.startswith('    5') or line.startswith('    6'):
            data = line.split()
            df = df.append({'mode': int(data[0]), 
                            'minimizedAffinity': float(data[1]), 
                            'CNNscore': float(data[2]), 
                            'CNNaffinity': float(data[3])}, 
                        ignore_index=True)
    return df

def get_center_coords_from_sdf(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    mol = suppl[0]
    if mol == None:
        mol2_path = sdf_path.replace(".sdf", ".mol2")
        mol = Chem.MolFromMol2File(mol2_path)
        if mol == None:
            lig_pdb = mol2_path.replace("_ligand.mol2", "_LIG.pdb")
            mol = Chem.MolFromPDBFile(lig_pdb, sanitize=True, removeHs=True) 
    coords = []
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    center = list(map(lambda x: sum(x)/len(x), zip(*coords)))
    
    return center

def get_center_coords_from_mol2(mol2_path):

    mol = Chem.MolFromMol2File(mol2_path, sanitize=False)
    if mol == None:
        sdf_path = mol2_path.replace(".mol2", ".sdf")
        mol = Chem.SDMolSupplier(sdf_path)[0]
        if mol == None:
            lig_pdb = mol2_path.replace("_ligand.mol2", "_LIG.pdb")
            mol = Chem.MolFromPDBFile(lig_pdb, sanitize=True, removeHs=True) 
    coords = []
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    center = list(map(lambda x: sum(x)/len(x), zip(*coords)))

    return center

def restore_pdb(packing_result, full_protein_path):
    parser = PDBParser()
    full_protein = parser.get_structure('FULL', full_protein_path)
    pocket_protein = parser.get_structure('POCKET', packing_result)
    residues = Selection.unfold_entities(full_protein, 'R')
    for residue in residues:
        residue_id = residue.get_id()
        chain_id = residue.get_parent().get_id()
        pocket_residues = pocket_protein[0][chain_id]
        if residue_id in pocket_residues:
            atoms = Selection.unfold_entities(residue, 'A')
            for atom in atoms:
                residue_atom_name = atom.get_name()
                if residue_atom_name[0] not in ['H', '1', '2', '3']:
                    pocket_atom_name = residue_atom_name.replace(' ', '')
                    pocket_atom = pocket_residues[residue_id][pocket_atom_name]
                    atom.set_coord(pocket_atom.get_coord())
    io = PDBIO()
    io.set_structure(full_protein)
    full_packing_path = packing_result.replace("new.pdb",".pdb")
    io.save(full_packing_path)

    return full_packing_path


def extract_glide_score(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
    mol = supplier[0]
    if mol == None:
        mol2_path = sdf_path.replace(".sdf", ".mol2")
        mol = Chem.MolFromMol2File(mol2_path)
        if mol == None:
            lig_pdb = mol2_path.replace("_ligand.mol2", "_LIG.pdb")
            mol = Chem.MolFromPDBFile(lig_pdb, sanitize=True, removeHs=True) 
    score = float(mol.GetProp('r_i_docking_score'))
    
    return score


def merge_docking_ligand_flie(file_names, indices, out_path):
    mols = []
    for i, file_name in enumerate(file_names):
        conformer_indices = indices
        suppl = Chem.SDMolSupplier(file_name)
        for j, mol in enumerate(suppl):
            if mol is not None and (i,j) in conformer_indices:
                mols.append(mol)
    ligand_pose_path = f"{out_path}/final_docking_top6.sdf"
    w = Chem.SDWriter(ligand_pose_path)
    for mol in mols:
        w.write(mol)
    w.close()

    return ligand_pose_path

def calculate_center(coordinates):
    total_x = 0.0
    total_y = 0.0
    total_z = 0.0

    num_coordinates = len(coordinates)
    for coord in coordinates:
        total_x += coord[0]
        total_y += coord[1]
        total_z += coord[2]
    center_x = total_x / num_coordinates
    center_y = total_y / num_coordinates
    center_z = total_z / num_coordinates

    return [center_x, center_y, center_z]


def get_pocket_outside_num(coords):
    ori_coords = [atom_coords for residue_coords in coords for atom_coords in residue_coords]
    center = calculate_center(ori_coords)
    center_max = [max(coord) for coord in zip(*ori_coords)]
    center_min = [min(coord) for coord in zip(*ori_coords)]
    size = [center_max[i] - center_min[i] for i in range(3)]
    inner_size = [a - a / 3 for a in size] #inference
    # inner_size = [a - a / 5 for a in size] #train
    inner_max = [center[i] + inner_size[i] / 2 for i in range(3)]
    inner_min = [center[i] - inner_size[i] / 2 for i in range(3)]
    res_num = []
    for residue_idx, residue_coords in enumerate(coords):
        for atom_coords in residue_coords:
            if any(a > inner_max[i] or a < inner_min[i] for i, a in enumerate(atom_coords)):
                res_num.append(residue_idx)
                break
    return res_num


def get_ligand_outside_num(coords,ligand_mol,cutoff):
    res_num = []
    for residue_idx, residue_coords in enumerate(coords):
        distance_list = []
        for atom_coords in residue_coords:
            distance = calculate_shortest_distance(ligand_mol,atom_coords)
            distance_list.append(distance)
        if min(distance_list) > cutoff:
            res_num.append(residue_idx)
    return res_num


def calculate_shortest_distance(ligand_mol, coordinate):
    coord = Point3D(coordinate[0], coordinate[1], coordinate[2])
    shortest_distance = float('inf')
    for atom in ligand_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_coord = ligand_mol.GetConformer().GetAtomPosition(atom_idx)
        distance = Point3D.Distance(atom_coord, coord)
        if distance < shortest_distance:
            shortest_distance = distance

    return shortest_distance
