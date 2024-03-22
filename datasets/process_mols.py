import copy
import os
import warnings

import numpy as np
import scipy.spatial as spa
import torch
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
from scipy import spatial
from scipy.special import softmax
from torch_cluster import radius_graph


import torch.nn.functional as F

from datasets.conformer_matching import get_torsion_angles, optimize_rotatable_bonds
from utils.torsion import get_transformation_mask


from Bio.PDB.vectors import calc_dihedral

import pandas as pd

biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 11)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 26)


def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return torch.tensor(atom_features_list)

def rec_atom_featurizer_from_rdkit(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
        ])

    return torch.tensor(atom_features_list)


def rec_residue_featurizer(rec):
    feature_list = []
    for residue in rec.get_residues():
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1



def parse_receptor(pdbid, pdbbind_dir):
    rec = parsePDB(pdbid, pdbbind_dir)
    return rec


def parsePDB(pdbid, pdbbind_dir):
    # rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')   #if used PDBBind_processed dataset
    rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein.pdb')   #if used side_chain dataset
    # rec_path = os.path.join(pdbbind_dir,pdbid)   #if used BC40 dataset
    return parse_pdb_from_path(rec_path)

def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', path)
        mol = Chem.MolFromPDBFile(path, sanitize=True, removeHs=True)
        rec = structure[0]
    return rec, mol


def extract_receptor_structure(rec, lig, lm_embedding_chains=None):

    """ Identify CYS containing disulfide bonds, pass the extract_rotamer function """
    cys_sulfurs = []
    disulfide_bond_res = []
    try:
        for residue in rec.get_residues():
            if residue.get_resname() == "CYS":
                sulfur = residue["SG"] # SG is the atomic name of cysteine sulfur atom
                cys_sulfurs.append(sulfur)
        ns = NeighborSearch(cys_sulfurs)
        disulfides = ns.search_all(3.0)  # Search for disulfide bonds within 3 angstroms
        cys_resnums = set()
        for disulfide in disulfides:
            cys1 = disulfide[0].get_parent().get_id()[1]  # The serial number of cysteine 1 residues
            cys2 = disulfide[1].get_parent().get_id()[1]  # The serial number of cysteine 2 residues
            cys_resnums.add(cys1)
            cys_resnums.add(cys2)
        for residue in rec.get_residues():
            if residue.get_id()[1] in cys_resnums:
                disulfide_bond_res.append(residue.get_id()[1])
    except:
        # print("This protein pocket has no SG atoms")
        pass
    
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    min_distances = []
    coords = []
    c_alpha_coords = []
    physicochemistry = []
    PSP19 = []

    Rotamer_idx = []
    Backbone_idx = []

    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_physicochemistry = []
        chain_PSP19 = []

        chain_Rotamer_idx = []
        chain_Backbone_idx = []

        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            AA_name = residue.get_resname()
            PSP19_chain, physicochemistry_chain = extract_fea(AA_name)


            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_rotamer_idx, residue_backbone_idx = extract_rotamer(residue,AA_name,disulfide_bond_res)
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_physicochemistry.append(physicochemistry_chain)
                chain_PSP19.append(PSP19_chain)
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                chain_Rotamer_idx.append(np.array(residue_rotamer_idx))
                chain_Backbone_idx.append(np.array(residue_backbone_idx))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
                print(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf

        min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))

        physicochemistry.append(np.array(chain_physicochemistry))
        PSP19.append(np.array(chain_PSP19))

        Rotamer_idx.append(chain_Rotamer_idx)
        Backbone_idx.append(chain_Backbone_idx)

        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        if not count == 0: valid_chain_ids.append(chain.get_id())

    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))
    valid_coords = []
    valid_c_alpha_coords = []
    valid_rotamer_idx = []
    valid_backbone_idx = []

    valid_PSP19 = []
    valid_physicochemistry = []


    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])

            valid_PSP19.append(PSP19[i])
            valid_physicochemistry.append(physicochemistry[i])
            valid_rotamer_idx.append(Rotamer_idx[i])
            valid_backbone_idx.append(Backbone_idx[i])


            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError('Encountered valid chain id that was not present in the LM embeddings')
                valid_lm_embeddings.append(lm_embedding_chains[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]
    rotamer_idx = [item for sublist in valid_rotamer_idx for item in sublist]
    backbone_idx = [item for sublist in valid_backbone_idx for item in sublist]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]

    physicochemistry = np.concatenate(valid_physicochemistry, axis=0)
    PSP19 = np.concatenate(valid_PSP19, axis=0)


    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    assert len(c_alpha_coords) == len(physicochemistry)
    assert len(c_alpha_coords) == len(PSP19)

    return rec, coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, lm_embeddings


def get_lig_graph(mol, complex_graph):
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph['ligand'].x = atom_feats
    complex_graph['ligand'].pos = lig_coords
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr
    return

def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)

def get_lig_graph_with_matching(mol_, complex_graph, popsize, maxiter, matching, keep_original, num_conformers, remove_hs):
    if matching:
        mol_maybe_noh = copy.deepcopy(mol_)
        if remove_hs:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True)
        if keep_original:
            complex_graph['ligand'].orig_pos = mol_maybe_noh.GetConformer().GetPositions()

        rotable_bonds = get_torsion_angles(mol_maybe_noh)
        if not rotable_bonds: print("no_rotable_bonds but still using it")

        for i in range(num_conformers):
            mol_rdkit = copy.deepcopy(mol_)

            mol_rdkit.RemoveAllConformers()
            mol_rdkit = AllChem.AddHs(mol_rdkit)
            generate_conformer(mol_rdkit)
            if remove_hs:
                mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
            mol = copy.deepcopy(mol_maybe_noh)
            if rotable_bonds:
                optimize_rotatable_bonds(mol_rdkit, mol, rotable_bonds, popsize=popsize, maxiter=maxiter)
            mol.AddConformer(mol_rdkit.GetConformer())
            rms_list = []
            AllChem.AlignMolConformers(mol, RMSlist=rms_list)
            mol_rdkit.RemoveAllConformers()
            mol_rdkit.AddConformer(mol.GetConformers()[1])

            if i == 0:
                complex_graph.rmsd_matching = rms_list[0]
                get_lig_graph(mol_rdkit, complex_graph)
            else:
                if torch.is_tensor(complex_graph['ligand'].pos):
                    complex_graph['ligand'].pos = [complex_graph['ligand'].pos]
                complex_graph['ligand'].pos.append(torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float())

    else:  # no matching
        complex_graph.rmsd_matching = 0
        
        if remove_hs: mol_ = Chem.RemoveHs(mol_)
        get_lig_graph(mol_, complex_graph)

    # edge_mask, mask_rotate = get_transformation_mask(complex_graph)
    # complex_graph['ligand'].edge_mask = torch.tensor(edge_mask)
    # complex_graph['ligand'].mask_rotate = mask_rotate

    return


def get_calpha_graph(rec, c_alpha_coords, Rotamer_0, Rotamer_1, Rotamer_2, Rotamer_3, physicochemistry, PSP19, n_coords, c_coords, complex_graph, cutoff=20, max_neighbor=None, lm_embeddings=None):
    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec)

    residue_list = ["GLY", "ALA", "SER", "CYS", "VAL", "THR", "PRO", "ILE", "LEU", "ASP", "ASN", "HIS", "PHE", "TYR", "TRP", "GLU", "GLN", "MET", "ARG", "LYS"]
    side_0_weight_true_list = [1 if residue.get_resname() in residue_list[2:] else 0 for residue in rec.get_residues()]
    side_1_weight_true_list = [1 if residue.get_resname() in residue_list[7:] else 0 for residue in rec.get_residues()]
    side_2_weight_true_list = [1 if residue.get_resname() in residue_list[15:] else 0 for residue in rec.get_residues()]
    side_3_weight_true_list = [1 if residue.get_resname() in residue_list[18:] else 0 for residue in rec.get_residues()]
    side_0_weight_true = np.array(side_0_weight_true_list)
    side_1_weight_true = np.array(side_1_weight_true_list)
    side_2_weight_true = np.array(side_2_weight_true_list)
    side_3_weight_true = np.array(side_3_weight_true_list)
    complex_graph.side_0_weight_true = side_0_weight_true
    complex_graph.side_1_weight_true = side_1_weight_true
    complex_graph.side_2_weight_true = side_2_weight_true
    complex_graph.side_3_weight_true = side_3_weight_true


    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))

    complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1) if lm_embeddings is not None else node_feat
    complex_graph['receptor'].pos = torch.from_numpy(c_alpha_coords).float()

    complex_graph['receptor'].rotamer_0 = torch.from_numpy(Rotamer_0).float()
    complex_graph['receptor'].rotamer_1 = torch.from_numpy(Rotamer_1).float()
    complex_graph['receptor'].rotamer_2 = torch.from_numpy(Rotamer_2).float()
    complex_graph['receptor'].rotamer_3 = torch.from_numpy(Rotamer_3).float()
    complex_graph['receptor'].physicochemistry = torch.from_numpy(physicochemistry).float()
    complex_graph['receptor'].PSP19 = torch.from_numpy(PSP19).float()


    complex_graph['receptor'].mu_r_norm = mu_r_norm
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    return


def rec_atom_featurizer(rec):
    atom_feats = []
    for i, atom in enumerate(rec.get_atoms()):
        atom_name, element = atom.name, atom.element
        if element == 'CD':
            element = 'C'
        assert not element == ''
        try:
            atomic_num = periodic_table.GetAtomicNumber(element)
        except:
            atomic_num = -1
        atom_feat = [safe_index(allowable_features['possible_amino_acids'], atom.get_parent().get_resname()),
                     safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                     safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                     safe_index(allowable_features['possible_atom_type_3'], atom_name)]
        atom_feats.append(atom_feat)

    return atom_feats


def get_rec_graph(rec, pdb_mol, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, pocket_outside_num, pocket_outside_idx, complex_graph, rec_radius, c_alpha_max_neighbors=None, all_atoms=False,
                  atom_radius=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None):
    if all_atoms:
        return get_fullrec_graph(rec, pdb_mol, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, pocket_outside_num, pocket_outside_idx, complex_graph,
                                 c_alpha_cutoff=rec_radius, c_alpha_max_neighbors=c_alpha_max_neighbors,
                                 atom_cutoff=atom_radius, atom_max_neighbors=atom_max_neighbors, remove_hs=remove_hs,lm_embeddings=lm_embeddings)
    else:
        return get_calpha_graph(rec, c_alpha_coords, rotamer_idx, physicochemistry, PSP19, n_coords, c_coords, complex_graph, rec_radius, c_alpha_max_neighbors,lm_embeddings=lm_embeddings)


def get_fullrec_graph(rec, pdb_mol, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physicochemistry, PSP19, n_coords, c_coords, pocket_outside_num, pocket_outside_idx, complex_graph, c_alpha_cutoff=20,
                      c_alpha_max_neighbors=None, atom_cutoff=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None):
    # builds the receptor graph with both residues and atoms

    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph of residues
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < c_alpha_cutoff)[0])
        dst.remove(i)
        if c_alpha_max_neighbors != None and len(dst) > c_alpha_max_neighbors:
            dst = list(np.argsort(distances[i, :]))[1: c_alpha_max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'The c_alpha_cutoff {c_alpha_cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert 1 - 1e-2 < weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))
    
    src_c_alpha_idx = np.concatenate([np.asarray([i]*len(l)) for i, l in enumerate(rec_coords)])
    atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec)))
    atom_coords = torch.from_numpy(np.concatenate(rec_coords, axis=0)).float()

    atom_num = 0
    src = []
    dst = []
    rotamer = []
    for residue_idx, residue_rotamer_idx in enumerate(rotamer_idx):

        for atom_rotamer_idx in residue_rotamer_idx:

            for i in range(1, 5):
                if atom_rotamer_idx[0] == i and residue_idx not in pocket_outside_num:
                    src.append(atom_num)
                    rotamer.append(atom_rotamer_idx[2])
                if atom_rotamer_idx[1] == i and residue_idx not in pocket_outside_num:
                    dst.append(atom_num)
            atom_num += 1
    
    tor_idx = torch.tensor(np.array([src, dst])).long()

    assert tor_idx.numel() != 0
    rotamer = torch.tensor(np.array(rotamer)).long()

    all_backbone_idx = torch.from_numpy(np.concatenate(backbone_idx, axis=0)).float()
    assert all_backbone_idx.shape[0] == atom_coords.shape[0],"backbone_atom"
    all_pocket_outside_idx = torch.from_numpy(np.concatenate(pocket_outside_idx, axis=0)).float()

    filter_idx = backbone_idx
    for i in pocket_outside_num:
        filter_idx[i] = np.ones_like(backbone_idx[i])
    all_filter_idx = np.concatenate(filter_idx, axis=0)

    # rotated_atom_num = atom_coords.shape[0] - all_backbone_idx.sum()   #Exclude RMSD of main chain atoms
    rotated_atom_num = atom_coords.shape[0] - all_filter_idx.sum()   #Exclude RMSD of main chain atoms and atoms with ligands greater than 7A away


    if remove_hs:
        not_hs = (atom_feat[:, 1] != 0)
        src_c_alpha_idx = src_c_alpha_idx[not_hs]
        atom_feat = atom_feat[not_hs]
        atom_coords = atom_coords[not_hs]

    atoms_edge_index = radius_graph(atom_coords, atom_cutoff, max_num_neighbors=atom_max_neighbors if atom_max_neighbors else 1000)
    atom_res_edge_index = torch.from_numpy(np.asarray([np.arange(len(atom_feat)), src_c_alpha_idx])).long()

    # AllChem.EmbedMolecule(pdb_mol)

    # Obtaining bond Information
    bond_index = []
    for bond in pdb_mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_index.append((begin_atom_idx, end_atom_idx))
    bond_idx = torch.tensor(bond_index)

    protein_atom_feature = rec_atom_featurizer_from_rdkit(pdb_mol)
    complex_graph['atom'].x = torch.cat([atom_feat,protein_atom_feature], axis=1)
    complex_graph['atom', 'atom_contact', 'atom'].edge_index = bond_idx.T
    all_bonds = bond_idx
    tor_bonds = tor_idx.T
    all_bonds_list = all_bonds.tolist()
    tor_bonds_list = tor_bonds.tolist()


    edge_mask, mask_rotate = get_transformation_mask(complex_graph,tor_idx,all_backbone_idx,all_pocket_outside_idx)
    tor_mask = np.asarray([1 if l  in tor_bonds_list or l[::-1] in tor_bonds_list else 0 for l in all_bonds_list], dtype=bool)   #There may be cases where the edge order is reversed in inference
    assert tor_mask.sum() == mask_rotate.shape[0],"out of bounds"
    complex_graph['atom'].pos = atom_coords
    complex_graph.edge_mask = torch.tensor(edge_mask)
    complex_graph.mask_rotate = mask_rotate

    physicochemistry_feat = torch.from_numpy(physicochemistry).float()
    PSP19_feat = torch.from_numpy(PSP19).float()
    complex_graph['receptor'].x = torch.cat([node_feat, physicochemistry_feat, PSP19_feat, torch.tensor(lm_embeddings)], axis=1) if lm_embeddings is not None else torch.cat([node_feat, physicochemistry_feat, PSP19_feat], axis=1)
    complex_graph['receptor'].pos = torch.from_numpy(c_alpha_coords).float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    complex_graph.tor_mask = torch.tensor(tor_mask)
    complex_graph.rotated_atom_num = rotated_atom_num

    complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index
    complex_graph['receptor'].mu_r_norm = mu_r_norm
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()


    return

def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    w.write(mol)
    w.close()

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.mol'):
        mol = Chem.MolFromMolFile(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol


def read_sdf_or_mol2(sdf_fileName, mol2_fileName):

    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            problem = False
        except Exception as e:
            problem = True

    return mol, problem



def extract_rotamer(residue,AA_name,disulfide_bond_res):
    idx_list = []
    src_idx = np.zeros(len(residue))
    dst_idx = np.zeros(len(residue))
    rotamer = np.zeros(len(residue))

    residue_AAname = []
    for i, atom in enumerate(residue):
        residue_AAname.append(atom.name)

    backbone_list = []
    backbone_idx = np.zeros(len(residue))
    for i, atom in enumerate(residue):
        if atom.name == 'N':
            backbone_idx[i] = 1
        elif atom.name == 'CA':
            backbone_idx[i] = 1
        elif atom.name == 'C':
            backbone_idx[i] = 1
        elif atom.name == 'O':
            backbone_idx[i] = 1
        else:
            pass
    if AA_name == "CYS" and residue.get_id()[1] in disulfide_bond_res:
        pass  #For CYS containing disulfide bonds, the side chain does not rotate
    elif AA_name == "GLY" or AA_name == "ALA":
        pass  # No changes needed
    elif AA_name == "PRO":
        pass  #For proline, the side chain does not rotate
    elif AA_name == "SER":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'OG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'OG' in residue_AAname :
                src_idx[i] = 0
                dst_idx[i] = 1
            else:
                pass
    elif AA_name == "CYS":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'SG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'SG' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 1
            else:
                pass
    elif AA_name == "VAL":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG1' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 1
            else:
                pass
    elif AA_name == "THR":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'OG1' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'OG1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 1
            else:
                pass
    elif AA_name == "ILE":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG1' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG1' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'CD1' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG1' and 'CD1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 2
            else:
                pass
    elif AA_name == "LEU":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'CD1' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'CD1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 2
            else:
                pass
    elif AA_name == "ASP" or AA_name == "ASN":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'OD1' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'OD1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 2
            else:
                pass
    elif AA_name == "HIS":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'ND1' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'ND1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 2
            else:
                pass
    elif AA_name == "PHE" or AA_name == "TYR" or AA_name == "TRP":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'CD1' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'CD1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 2
            else:
                pass
    elif AA_name == "GLU" or AA_name == "GLN":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'CD' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'CD' in residue_AAname:
                dst_idx[i] = 2
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CG' and 'OE1' in residue_AAname:
                src_idx[i] = 3
            elif atom.name == 'CD' and 'OE1' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 3
            else:
                pass
    elif AA_name == "MET":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'SD' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'SD' in residue_AAname:
                dst_idx[i] = 2
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CG' and 'CE' in residue_AAname:
                src_idx[i] = 3
            elif atom.name == 'SD' and 'CE' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 3
            else:
                pass
    elif AA_name == "ARG":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'CD' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'CD' in residue_AAname:
                dst_idx[i] = 2
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CG' and 'NE' in residue_AAname:
                src_idx[i] = 3
            elif atom.name == 'CD' and 'NE' in residue_AAname:
                dst_idx[i] = 3
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CD' and 'CZ' in residue_AAname:
                src_idx[i] = 4
            elif atom.name == 'NE' and 'CZ' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 4
            else:
                pass
    elif AA_name == "LYS":
        for i, atom in enumerate(residue):
            if atom.name == 'CA' and 'CG' in residue_AAname:
                src_idx[i] = 1
                dst_idx[i] = 0
            elif atom.name == 'CB' and 'CG' in residue_AAname:
                dst_idx[i] = 1
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CB' and 'CD' in residue_AAname:
                src_idx[i] = 2
            elif atom.name == 'CG' and 'CD' in residue_AAname:
                dst_idx[i] = 2
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CG' and 'CE' in residue_AAname:
                src_idx[i] = 3
            elif atom.name == 'CD' and 'CE' in residue_AAname:
                dst_idx[i] = 3
            else:
                pass
        for i, atom in enumerate(residue):
            if atom.name == 'CD' and 'NZ' in residue_AAname:
                src_idx[i] = 4
            elif atom.name == 'CE' and 'NZ' in residue_AAname:
                src_idx[i] = 0
                dst_idx[i] = 4
            else:
                pass
    else:
        # raise ValueError("Unsupported amino acid: {}".format(AA_name))
        # print("Unsupported amino acid: {}".format(AA_name))
        pass
    while np.count_nonzero(src_idx) != np.count_nonzero(dst_idx):   #When only one of the atoms of the possible rotating bond is identified, the rotating bond is not rotated
        if np.count_nonzero(src_idx) > np.count_nonzero(dst_idx):
            last_nonzero_index = np.argwhere(src_idx)[::-1][0][0]
            src_idx[last_nonzero_index] = 0
            # print("here")
            # print(AA_name)
        else:
            last_nonzero_index = np.argwhere(dst_idx)[::-1][0][0]
            dst_idx[last_nonzero_index] = 0

    if dst_idx[-1] != 0:   #When the rotatable key is the end key, it is not rotated
        dst_idx[-1] = 0
        last_nonzero_index = np.argwhere(src_idx)[::-1][0][0]
        src_idx[last_nonzero_index] = 0

    idx = np.concatenate((src_idx[:, np.newaxis], dst_idx[:, np.newaxis], rotamer[:, np.newaxis]), axis=1)
    idx_list = idx.tolist()
    backbone_list = backbone_idx.tolist()
    # print("idx_list",idx_list)
    return idx_list, backbone_list

# def extract_rotamer(residue, AA_name, disulfide_bond_res):
#     rotamers = {
#         "SER": [("CA", "CB"), ("CB",)],
#         "CYS": [("CA", "CB"), ("CB",)],
#         "VAL": [("CA", "CB"), ("CB",)],
#         "THR": [("CA", "CB"), ("CB",)],
#         "ILE": [("CA", "CB", "CG1"), ("CB", "CG1"), ("CG1",)],
#         "LEU": [("CA", "CB", "CG"), ("CB", "CG"), ("CG",)],
#         "ASP": [("CA", "CB", "CG"), ("CB", "CG"), ("CG",)],
#         "ASN": [("CA", "CB", "CG"), ("CB", "CG"), ("CG",)],
#         "HIS": [("CA", "CB", "CG"), ("CB", "CG"), ("CG",)],
#         "PHE": [("CA", "CB", "CG"), ("CB", "CG"), ("CG",)],
#         "TYR": [("CA", "CB", "CG"), ("CB", "CG"), ("CG",)],
#         "TRP": [("CA", "CB", "CG"), ("CB", "CG"), ("CG",)],
#         "GLU": [("CA", "CB", "CG", "CD"), ("CB", "CG", "CD"), ("CG", "CD"), ("CD",)],
#         "GLN": [("CA", "CB", "CG", "CD"), ("CB", "CG", "CD"), ("CG", "CD"), ("CD",)],
#         "MET": [("CA", "CB", "CG", "SD"), ("CB", "CG", "SD"), ("CG", "SD"), ("SD",)],
#         "ARG": [("CA", "CB", "CG", "CD", "NE"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE"), ("CD", "NE"), ("NE",)],
#         "LYS": [("CA", "CB", "CG", "CD", "CE"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE"), ("CD", "CE"), ("CE",)],
#     }

#     idx_list = []
#     backbone_list = []
#     atom_names = [atom.name for atom in residue]

#     for i, atom in enumerate(residue):
#         if atom.name == 'N' or atom.name == 'CA' or atom.name == 'C' or atom.name == 'O':
#             backbone_list.append(1)
#         else:
#             backbone_list.append(0)

#     if AA_name == "CYS" and residue.get_id()[1] in disulfide_bond_res:
#         return idx_list, backbone_list

#     if AA_name in rotamers:
#         for rotamer in rotamers[AA_name]:
#             src_idx = [0] * len(residue)
#             dst_idx = [0] * len(residue)
#             rotamer_indices = []

#             for atom_name in rotamer:
#                 atom_index = atom_names.index(atom_name)
#                 src_idx[atom_index] = 1
#                 dst_idx[atom_index] = 0
#                 rotamer_indices.append(atom_index)

#             while len(rotamer_indices) > 1:
#                 last_nonzero_index = rotamer_indices.pop()
#                 src_idx[last_nonzero_index] = 0

#             if len(rotamer_indices) == 1:
#                 last_nonzero_index = rotamer_indices[0]
#                 src_idx[last_nonzero_index] = 0
#                 dst_idx[last_nonzero_index] = 0

#             idx = list(zip(src_idx, dst_idx, [0] * len(residue)))
#             idx_list.extend(idx)

#     return idx_list, backbone_list

standard_aa_names = {
                   "ALA":0,
                   "CYS":1,
                   "ASP":2,
                   "GLU":3,
                   "PHE":4,
                   "GLY":5,
                   "HIS":6,
                   "ILE":7,
                   "LYS":8,
                   "LEU":9,
                   "MET":10,
                   "ASN":11,
                   "PRO":12,
                   "GLN":13,
                   "ARG":14,
                   "SER":15,
                   "THR":16,
                   "VAL":17,
                   "TRP":18,
                   "TYR":19,
                   }
physicochemistry = pd.read_csv('./data/physicochemisty',sep='\s+',header=None)
PSP19 = np.loadtxt("./data/PSP19")

def extract_fea(AA_name):
    try:
        AA_label = standard_aa_names[AA_name]
        PSP19_chain = PSP19[AA_label]
        physicochemistry_chain = physicochemistry.iloc[:, 1:].values[AA_label]
    except:
        PSP19_chain = PSP19[20]
        physicochemistry_chain = physicochemistry.iloc[:, 1:].values[20]
    return PSP19_chain, physicochemistry_chain