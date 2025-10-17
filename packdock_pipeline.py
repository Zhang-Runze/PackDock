#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PackDock docking pipeline.

Stages:
0. prepare & pocket extraction
1. protein ligand-free side-chain packing 
2. docking on packed structures
3. protein ligand-conditioned side-chain packing 
4. docking on packed structures

"""

from __future__ import annotations

import os
import re
import gc
import json
import copy
import shutil
import warnings
import subprocess
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, RDLogger
from Bio.PDB import PDBParser, PDBIO, Select
from torch_geometric.data import HeteroData

# local project modules (assumed present)
from datasets.pdbbind import PDBBind, read_mol, read_pdb
from datasets.process_mols import (
    parse_pdb_from_path, read_molecule, extract_receptor_structure,
    get_rec_graph, get_lig_graph_with_matching
)
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import (
    docking_with_vina, get_model,
    pdb2pdbqt, read_docking_log, get_pocket_outside_num, get_ligand_outside_num
)
from utils.visualise import PDBFile

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.DisableLog("rdApp.*")

PACKDOCK_DIR = os.environ.get(
    "PACKDOCK_DIR",
    os.path.dirname(__file__)
)

Entry = Dict[str, Any]


# ---------------------------
# CLI
# ---------------------------
def parse_arguments() -> Namespace:
    p = ArgumentParser(
        description="PackDock pipeline",
        formatter_class=RawDescriptionHelpFormatter
    )

    # Model configuration
    p.add_argument("--protein_model_dir", type=str, default="workdir/protein_model")
    p.add_argument("--model_dir", type=str, default="workdir/complex_model")
    p.add_argument("--ckpt", type=str, default="best_ema_inference_epoch_model.pt")

    # Sampling
    p.add_argument("--inference_steps", type=int, default=20)
    p.add_argument("--actual_steps", type=int, default=20)
    p.add_argument("--no_random", action="store_true", default=False)
    p.add_argument("--ode", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=36)
    p.add_argument("--no_final_step_noise", action="store_true", default=False)

    # Speed / batching
    p.add_argument("--pack_group_size", type=int, default=1)
    p.add_argument("--graph_build_workers", type=int, default=0)

    # Docking
    p.add_argument("--docking_max_workers", type=int, default=1)
    p.add_argument("--docking_max_inflight", type=int, default=None)
    p.add_argument("--vina_path", type=str, default="./vina")

    # Receptor parameters
    p.add_argument("--receptor_radius", type=float, default=15)
    p.add_argument("--c_alpha_max_neighbors", type=int, default=24)
    p.add_argument("--all_atoms", action="store_true", default=True)
    p.add_argument("--atom_radius", type=float, default=5)
    p.add_argument("--atom_max_neighbors", type=int, default=8)
    p.add_argument("--remove_hs", action="store_true", default=True)

    # Matching
    p.add_argument("--matching_popsize", type=int, default=20)
    p.add_argument("--matching_maxiter", type=int, default=20)
    p.add_argument("--num_conformers", type=int, default=1)

    # I/O
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results/test")
    p.add_argument("--seed", type=str, default="666")
    p.add_argument("--save_visualisation", action="store_true", default=True)
    p.add_argument("--rmsd", action="store_true", default=False)

    # Pocket extraction
    p.add_argument("--extract_pocket", action="store_true", default=True)
    p.add_argument("--pocket_radius", type=float, default=5.0)
    p.add_argument("--include_backbone", action="store_true", default=True)
    p.add_argument("--min_pocket_radius", type=int, default=3)

    # Packing
    p.add_argument("--stage1_packing_num", type=int, default=1)
    p.add_argument("--stage3_packing_num", type=int, default=5)

    return p.parse_args()


# ---------------------------
# Manifest manager
# ---------------------------
class ManifestManager:
    @staticmethod
    def path(out_dir: str, cid: str, stage: str, part: str) -> Path:
        return Path(out_dir) / f"{cid}__{stage}__{part}.json"

    @staticmethod
    def save(out_dir: str, cid: str, stage: str, part: str, payload: Dict[str, Any]):
        p = ManifestManager.path(out_dir, cid, stage, part)
        try:
            p.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            print(f"[{cid}] WARN: cannot write manifest {p}: {e}")

    @staticmethod
    def load(out_dir: str, cid: str, stage: str, part: str) -> Dict[str, Any]:
        p = ManifestManager.path(out_dir, cid, stage, part)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception as e:
            print(f"[{cid}] WARN: cannot read manifest {p}: {e}")
            return {}


# ---------------------------
# File utilities & checks
# ---------------------------
def copy_file(src: str, dst_dir: str) -> str:
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    out = Path(dst_dir) / Path(src).name
    shutil.copy(src, out)
    return str(out)

def run_obabel_inplace_pdb(pdb_path: str) -> bool:
    if not shutil.which("obabel"):
        print(f"[normalize] WARN: obabel not in PATH; skip {pdb_path}")
        return False
    try:
        subprocess.run(["obabel", "-ipdb", pdb_path, "-O", pdb_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[normalize] WARN: obabel failed on {pdb_path}: {e}")
        return False

def rdkit_can_read_pdb(pdb_path: str) -> bool:
    try:
        mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=True)
        return mol is not None
    except Exception:
        return False


# ---------------------------
# Stage 0: pocket extraction
# ---------------------------
def ligand_coords_from_any(ligand_path: str) -> Optional[np.ndarray]:
    try:
        lp = ligand_path.lower()
        if lp.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(ligand_path, sanitize=False, removeHs=False)
        elif lp.endswith(".sdf"):
            mol = Chem.SDMolSupplier(ligand_path, sanitize=False, removeHs=False)[0]
        elif lp.endswith(".mol2"):
            mol = Chem.MolFromMol2File(ligand_path, sanitize=False, removeHs=False)
        elif lp.endswith(".mol"):
            mol = Chem.MolFromMolFile(ligand_path, sanitize=False, removeHs=False)
        else:
            mol = Chem.MolFromPDBFile(ligand_path, sanitize=False, removeHs=False)
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            return np.array([[conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z]
                             for i in range(mol.GetNumAtoms())])
    except Exception as e:
        print(f"[ligand] RDKit failed on {ligand_path}: {e}")

    try:
        if ligand_path.lower().endswith(".pdb"):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("ligand", ligand_path)
            coords = [atom.get_coord() for atom in structure.get_atoms()]
            if coords:
                return np.array(coords)
    except Exception as e:
        print(f"[ligand] BioPython failed on {ligand_path}: {e}")
    return None

def candidate_radii(base: int, deltas=(0, -1, +1, -2, +2, -3, +3), min_radius: int = 2) -> List[int]:
    seen, out = set(), []
    for d in deltas:
        r = base + d
        if r >= min_radius and r not in seen:
            out.append(r); seen.add(r)
    return out

def extract_protein_pocket(protein_path: str, ligand_path: str, pocket_radius: float = 5.0,
                           output_path: Optional[str] = None, include_backbone: bool = True) -> str:
    if output_path is None:
        p = Path(protein_path).with_suffix("")
        l = Path(ligand_path).stem
        output_path = str(Path(protein_path).parent / f"{p.name}_{l}_pocket_{int(pocket_radius)}A.pdb")
    try:
        parser = PDBParser(QUIET=True)
        protein_structure = parser.get_structure("protein", protein_path)
        lig_coords = ligand_coords_from_any(ligand_path)
        if lig_coords is None:
            print(f"[pocket] WARN: no ligand coords, use original protein: {protein_path}")
            return protein_path
    except Exception as e:
        print(f"[pocket] ERROR: parse failed -> {e}; use original: {protein_path}")
        return protein_path

    class PocketSelect(Select):
        def accept_residue(self, residue) -> bool:
            residue_coords = []
            for atom in residue:
                if include_backbone or atom.get_name() not in ["N", "CA", "C", "O"]:
                    residue_coords.append(atom.get_coord())
            if not residue_coords:
                return False
            rc = np.asarray(residue_coords, dtype=float)
            d = np.linalg.norm(rc[:, None, :] - lig_coords[None, :, :], axis=2)
            return float(np.min(d)) <= float(pocket_radius)

    try:
        io = PDBIO()
        io.set_structure(protein_structure)
        io.save(output_path, PocketSelect())
        if Path(output_path).exists() and "ATOM" in Path(output_path).read_text():
            run_obabel_inplace_pdb(output_path)
            if rdkit_can_read_pdb(output_path):
                return output_path
    except Exception as e:
        print(f"[pocket] WARN: failed to write/validate pocket at {pocket_radius}Å -> {e}")
    return protein_path

def try_extract_pocket_robust(entry: Entry, args: Namespace) -> Entry:
    cid = entry["complex_id"]; prot = entry["protein_path"]; lig = entry["ligand_path"]; out_dir = entry["out_dir"]
    if not Path(prot).exists():
        print(f"[{cid}] WARN: protein not found -> {prot}"); entry["working_protein"] = prot; return entry
    if not Path(lig).exists():
        print(f"[{cid}] WARN: ligand not found  -> {lig}"); entry["working_protein"] = prot; return entry
    if not getattr(args, "extract_pocket", False):
        entry["working_protein"] = prot
        ManifestManager.save(out_dir, cid, "stage0", "postprocess", {
            "working_protein": prot, "pocket_radius_used": None, "pocket_attempts_used": 0,
            "protein_path": prot, "ligand_path": lig
        })
        return entry

    base = int(getattr(args, "pocket_radius", 7))
    min_r = int(getattr(args, "min_pocket_radius", 3))
    inc_bb = bool(getattr(args, "include_backbone", False))
    radii = candidate_radii(base, (0, -1, +1, -2, +2, -3, +3), min_r)

    for i, rad in enumerate(radii, start=1):
        out = str(Path(out_dir) / f"{cid}_pocket_{rad}A.pdb")
        try:
            pocket = extract_protein_pocket(prot, lig, rad, out, inc_bb)
            if rdkit_can_read_pdb(pocket):
                entry.update({"working_protein": pocket, "pocket_radius_used": rad, "pocket_attempts_used": i})
                ManifestManager.save(out_dir, cid, "stage0", "postprocess", {
                    "working_protein": pocket, "pocket_radius_used": rad, "pocket_attempts_used": i,
                    "protein_path": prot, "ligand_path": lig
                })
                return entry
        except Exception as e:
            print(f"[{cid}] WARN: pocket at {rad}Å failed -> {e}")

    print(f"[{cid}] WARN: all pocket attempts failed; fallback to original: {prot}")
    entry.update({"working_protein": prot, "pocket_radius_used": None, "pocket_attempts_used": len(radii)})
    ManifestManager.save(out_dir, cid, "stage0", "postprocess", {
        "working_protein": prot, "pocket_radius_used": None, "pocket_attempts_used": len(radii),
        "protein_path": prot, "ligand_path": lig
    })
    return entry

def read_input_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    required = ["protein_path", "ligand_path", "complex_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=required)
    if df.empty:
        raise ValueError("No valid rows in input CSV.")
    print(f"Loaded {len(df)} entries from {csv_file}")
    return df

def prepare_entries(args: Namespace) -> List[Entry]:
    df = read_input_csv(args.input_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Entry] = []
    for _, row in df.iterrows():
        cid = str(row["complex_id"]); odir = out_dir / cid; odir.mkdir(parents=True, exist_ok=True)
        entries.append({"complex_id": cid, "protein_path": str(row["protein_path"]),
                        "ligand_path": str(row["ligand_path"]), "out_dir": str(os.path.abspath(odir))})
    return entries

def stage0_prepare_and_pocket(args: Namespace) -> List[Entry]:
    entries = prepare_entries(args)
    print(f"\n▶ Stage 0 — Pocket Extraction for {len(entries)} entries")
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(entries)))) as ex:
        futs = [ex.submit(try_extract_pocket_robust, e, args) for e in entries]
        entries = [f.result() for f in as_completed(futs)]
    entries.sort(key=lambda d: d["complex_id"])
    return entries


# ---------------------------
# Graph construction
# ---------------------------
def construct_protein_graph(protein_path: str, pdb_id: str, ligand: Any, args: Namespace) -> List[HeteroData]:
    protein_graphs: List[HeteroData] = []
    rec_model, pdb_mol = parse_pdb_from_path(protein_path)
    protein_graph = HeteroData(); protein_graph["name"] = str(pdb_id)
    rec, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physchem, PSP19, n_coords, c_coords, lm = \
        extract_receptor_structure(copy.deepcopy(rec_model), ligand)
    pocket_outside_num = get_pocket_outside_num(rec_coords)
    pocket_outside_idx = [np.ones_like(arr) if i in pocket_outside_num else arr
                          for i, arr in enumerate(backbone_idx)]
    get_rec_graph(rec, pdb_mol, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx,
                  physchem, PSP19, n_coords, c_coords, pocket_outside_num, pocket_outside_idx,
                  protein_graph, rec_radius=args.receptor_radius,
                  c_alpha_max_neighbors=args.c_alpha_max_neighbors, all_atoms=args.all_atoms,
                  atom_radius=args.atom_radius, atom_max_neighbors=args.atom_max_neighbors,
                  remove_hs=args.remove_hs, lm_embeddings=lm)
    center = torch.mean(protein_graph["receptor"].pos, dim=0, keepdim=True)
    protein_graph["receptor"].pos -= center
    protein_graph["atom"].pos -= center
    protein_graph.original_center = center
    protein_graphs.append(protein_graph)
    return protein_graphs

def _read_ligand_state_any(pdbqt_path: str) -> Optional[Any]:
    for ext in (".pdbqt", ".mol", ".sdf", ".mol2"):
        try:
            path_ext = pdbqt_path if ext == ".pdbqt" else pdbqt_path.replace(".pdbqt", ext)
            if ext != ".pdbqt":
                os.system(f"obabel -ipdbqt {pdbqt_path} -O {path_ext}")
            lig = read_molecule(path_ext, remove_hs=False, sanitize=False)
            if ext != ".pdbqt" and Path(path_ext).exists():
                Path(path_ext).unlink(missing_ok=True)
            if lig is not None:
                return lig
        except Exception:
            continue
    return None

def get_packing_num(state_path: str) -> int:
    stem = Path(state_path).stem
    patterns = [
        r'(?:^|[_-])sample-(\d+)(?!.*(?:^|[_-])sample-\d+)',
        r'(?:^|[_-])model[_-]?(\d+)(?!.*(?:^|[_-])model[_-]?\d+)',
        r'(?:^|[_-])state[_-]?(\d+)(?!.*(?:^|[_-])state[_-]?\d+)',
        r'(?:^|[_-])pack(?:ing)?[_-]?(\d+)(?!.*(?:^|[_-])pack(?:ing)?[_-]?\d+)',
        r'(?:^|[_-])seed-(\d+)(?!.*(?:^|[_-])seed-\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            return int(m.group(1)) + 1
    nums = re.findall(r'(\d+)', stem)
    if nums:
        return int(nums[-1]) + 1
    raise ValueError(f"Cannot parse packing_num: {stem}")

def construct_complex_graphs(ligand_states: List[str], ori_protein_path: str, out_dir: str, pdb_id: str,
                             args: Namespace, loop_num: int) -> Tuple[List[HeteroData], str]:
    graphs: List[HeteroData] = []
    pdb_path_used = ""
    for idx, state_path in enumerate(ligand_states):
        try:
            packing_num = get_packing_num(state_path)
            pdb_path_used = f"{out_dir}/packing_{pdb_id}_{pdb_id}_{packing_num}new.pdb"
            lig = _read_ligand_state_any(state_path)
            if lig is None:
                print(f"[{pdb_id}] WARN: cannot read ligand state: {state_path}"); continue
            rec_model, pdb_mol = parse_pdb_from_path(ori_protein_path)
            data = HeteroData(); data["name"] = f"{loop_num}_{idx+1}"
            rec, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx, physchem, PSP19, n_coords, c_coords, lm = \
                extract_receptor_structure(copy.deepcopy(rec_model), lig)
            cut_off = 3.5
            ligand_outside_num = get_ligand_outside_num(rec_coords, lig, cut_off)
            ligand_outside_idx = [np.ones_like(arr) if i in ligand_outside_num else arr
                                  for i, arr in enumerate(backbone_idx)]
            get_rec_graph(rec, pdb_mol, rec_coords, c_alpha_coords, rotamer_idx, backbone_idx,
                          physchem, PSP19, n_coords, c_coords, ligand_outside_num, ligand_outside_idx,
                          data, rec_radius=args.receptor_radius,
                          c_alpha_max_neighbors=args.c_alpha_max_neighbors, all_atoms=args.all_atoms,
                          atom_radius=args.atom_radius, atom_max_neighbors=args.atom_max_neighbors,
                          remove_hs=args.remove_hs, lm_embeddings=lm)
            get_lig_graph_with_matching(lig, data, args.matching_popsize, args.matching_maxiter,
                                        False, True, args.num_conformers, remove_hs=args.remove_hs)
            center = torch.mean(data["receptor"].pos, dim=0, keepdim=True)
            data["receptor"].pos -= center; data["atom"].pos -= center; data["ligand"].pos -= center
            data.original_center = center
            graphs.append(data)
        except Exception as e:
            print(f"[{pdb_id}] WARN: invalid packing conformation; skip {pdb_path_used} -> {e}")
            continue
    return graphs, pdb_path_used

def _build_protein_graph_single(entry: dict, args: Namespace) -> Tuple[str, List[HeteroData]]:
    cid = entry["complex_id"]; wprot = entry.get("working_protein", entry["protein_path"]); lig_path = entry["ligand_path"]
    try:
        if lig_path.endswith(".pdb"):
            lig = Chem.MolFromPDBFile(lig_path, sanitize=True, removeHs=True)
        elif lig_path.endswith(".sdf"):
            lig = Chem.SDMolSupplier(lig_path, sanitize=True, removeHs=True)[0]
        else:
            lig = Chem.MolFromPDBFile(lig_path, sanitize=True, removeHs=True)
        graphs = construct_protein_graph(wprot, cid, lig, args)
        return cid, graphs
    except Exception as e:
        print(f"[{cid}] build protein graph failed: {e}")
        return cid, []

def _build_complex_graphs_single(entry: dict, args: Namespace) -> Tuple[str, List[HeteroData]]:
    cid = entry["complex_id"]; out_dir = entry["out_dir"]; wprot = entry.get("working_protein", entry["protein_path"])
    top_states = entry.get("stage2_top_ligand_states", [])
    if not top_states:
        return cid, []
    try:
        graphs, _ = construct_complex_graphs(top_states, wprot, out_dir, cid, args, loop_num=1)
        return cid, graphs
    except Exception as e:
        print(f"[{cid}] build complex graphs failed: {e}")
        return cid, []

# ---------------------------
# Packing core (shared)
# ---------------------------
def _pack_batch_and_write(owners: List[dict], orig_graphs: List[HeteroData], copies_per_graph: int,
                          model: Any, device: torch.device, side_schedule: Any, t_to_sigma: Any,
                          model_args: Namespace, args: Namespace) -> List[str]:
    data_list: List[HeteroData] = []; owner_index: List[int] = []
    for i, g in enumerate(orig_graphs):
        for _ in range(copies_per_graph):
            data_list.append(copy.deepcopy(g)); owner_index.append(i)
    if not data_list: return []
    randomize_position(data_list, False, False)
    vis_list = None
    if args.save_visualisation:
        vis_list = []
        for idx, sample in enumerate(data_list):
            e = owners[owner_index[idx]]
            ori_prot = e.get("working_protein", e["protein_path"])
            mol = Chem.MolFromPDBFile(ori_prot, sanitize=True, removeHs=True)
            pdb = PDBFile(mol); pdb.add(mol, 0, 0)
            pdb.add((sample["atom"].pos + sample.original_center).detach().cpu(), 1, 0)
            vis_list.append(pdb)
    with torch.no_grad():
        data_list, _ = sampling(
            data_list=data_list, model=model,
            inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
            side_schedule=side_schedule, device=device, t_to_sigma=t_to_sigma,
            model_args=model_args, no_random=args.no_random, ode=args.ode,
            visualization_list=vis_list, confidence_model=None, confidence_data_list=None,
            confidence_model_args=None, batch_size=args.batch_size,
            no_final_step_noise=args.no_final_step_noise
        )
    written: List[str] = []; local_counts: Dict[str, int] = {}
    for idx, sample in enumerate(data_list):
        e = owners[owner_index[idx]]; cid = e["complex_id"]
        local_counts.setdefault(cid, 0); local_counts[cid] += 1
        out_dir = e["out_dir"]
        out_pdb = f"{out_dir}/packing_{sample['name']}_{cid}_{local_counts[cid]}.pdb"
        if args.save_visualisation and vis_list is not None:
            vis = vis_list[idx]; vis.add((sample["atom"].pos + sample.original_center).detach().cpu(), part=1, order=1)
            vis.write(out_pdb)
        else:
            ori_prot = e.get("working_protein", e["protein_path"])
            mol = Chem.MolFromPDBFile(ori_prot, sanitize=True, removeHs=True)
            pdb = PDBFile(mol); pdb.add(mol, 0, 0)
            pdb.add((sample["atom"].pos + sample.original_center).detach().cpu(), 1, 0)
            pdb.write(out_pdb)
        written.append(out_pdb)
    _del_and_gc(data_list, vis_list)
    return written

# ---------------------------
# Models loading
# ---------------------------
def load_protein_model(args) -> Tuple[Any, torch.device, Any, Namespace]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(f"{args.protein_model_dir}/model_parameters.yml") as f:
        prot_args = Namespace(**yaml.full_load(f))
    t2s = partial(t_to_sigma_compl, args=prot_args)
    model = get_model(prot_args, device, t_to_sigma=t2s, no_parallel=True, only_protein=True)
    state = torch.load(f"{args.protein_model_dir}/{args.ckpt}", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device, t2s, prot_args

def load_complex_model(args) -> Tuple[Any, torch.device, Any, Namespace]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(f"{args.model_dir}/model_parameters.yml") as f:
        cmp_args = Namespace(**yaml.full_load(f))
    t2s = partial(t_to_sigma_compl, args=cmp_args)
    model = get_model(cmp_args, device, t_to_sigma=t2s, no_parallel=True)
    state = torch.load(f"{args.model_dir}/{args.ckpt}", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device, t2s, cmp_args

# ---------------------------
# GPU helpers
# ---------------------------
def _cuda_gc():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def _del_and_gc(*objs):
    for o in objs:
        try: del o
        except Exception: pass
    _cuda_gc()

def available_gpu_ids() -> List[int]:
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

def chunk_round_robin(items: List[Any], ngpu: int) -> List[List[Any]]:
    if ngpu <= 0:
        return [items]
    chunks = [[] for _ in range(ngpu)]
    for i, it in enumerate(items):
        chunks[i % ngpu].append(it)
    return chunks

def run_per_gpu(entries: List[Entry], worker, args: Namespace) -> List[Entry]:
    gpus = available_gpu_ids()
    if not gpus:
        return worker(entries, args, 0)
    chunks = chunk_round_robin(entries, len(gpus))
    results: List[Entry] = []
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(worker, ch, args, gid) for gid, ch in zip(gpus, chunks)]
        for f in as_completed(futs):
            results.extend(f.result())
    results.sort(key=lambda d: d["complex_id"])
    return results

# ---------------------------
# Stage 1 (packing)
# ---------------------------
def stage1_preprocess_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    protein_model, protein_device, protein_t2s, prot_args = load_protein_model(args)
    side_schedule = get_t_schedule(inference_steps=args.inference_steps)

    for e in chunk:
        ManifestManager.save(e["out_dir"], e["complex_id"], "stage1", "preprocess", {
            "model_dir": args.protein_model_dir, "ckpt": args.ckpt,
            "inference_steps": args.inference_steps, "batch_size": args.batch_size
        })

    group_size = max(1, int(getattr(args, "pack_group_size", 12)))
    for gstart in range(0, len(chunk), group_size):
        group = chunk[gstart:gstart + group_size]
        owners: List[dict] = []; graphs: List[HeteroData] = []
        if getattr(args, "graph_build_workers", 0) > 0 and len(group) > 1:
            with ThreadPoolExecutor(max_workers=args.graph_build_workers) as ex:
                futs = [ex.submit(_build_protein_graph_single, e, args) for e in group]
                for f in as_completed(futs):
                    cid, gs = f.result()
                    if gs:
                        e = next(e for e in group if e["complex_id"] == cid)
                        owners.append(e); graphs.extend(gs)
        else:
            for e in group:
                cid, gs = _build_protein_graph_single(e, args)
                if gs: owners.append(e); graphs.extend(gs)
        if not graphs: continue
        written = _pack_batch_and_write(owners=owners, orig_graphs=graphs, copies_per_graph=int(getattr(args, "stage1_packing_num", 1)),
                                       model=protein_model, device=protein_device,
                                       side_schedule=side_schedule, t_to_sigma=protein_t2s,
                                       model_args=prot_args, args=args)
        by_cid: Dict[str, List[str]] = {}
        for p in written:
            cid = Path(p).parent.name; by_cid.setdefault(cid, []).append(p)
        for e in owners:
            e["stage1_packed_proteins"] = by_cid.get(e["complex_id"], [])
            old = ManifestManager.load(e["out_dir"], e["complex_id"], "stage1", "core")
            cur = old.get("packed_proteins", []); cur.extend(e["stage1_packed_proteins"])
            ManifestManager.save(e["out_dir"], e["complex_id"], "stage1", "core", {"packed_proteins": cur})
        _del_and_gc(graphs, owners, written)

    _del_and_gc(protein_model, protein_device, protein_t2s, prot_args, side_schedule)
    return chunk

def stage1_postprocess(entries: List[Entry], args: Namespace) -> List[Entry]:
    for e in entries:
        # ensure stage1_packed_proteins exists in memory (or load from manifest)
        if not e.get("stage1_packed_proteins"):
            core = ManifestManager.load(e["out_dir"], e["complex_id"], "stage1", "core")
            e["stage1_packed_proteins"] = core.get("packed_proteins", [])
        # write a simple postprocess manifest that references packed_proteins
        ManifestManager.save(e["out_dir"], e["complex_id"], "stage1", "postprocess", {
            "packed_proteins": e["stage1_packed_proteins"]
        })
    return entries

# ---------------------------
# Stage 2
# ---------------------------
def _short(msg: str, maxlen: int = 180) -> str:
    s = str(msg); return s if len(s) <= maxlen else (s[: maxlen - 3] + "...")

def read_rmsd_from_txt(txt_path: str) -> Optional[float]:
    try:
        for line in Path(txt_path).read_text().splitlines():
            if line.startswith("RMSD"):
                return float(line.strip().split()[-1])
    except Exception as e:
        print(f"[rmsd] WARN: cannot parse {txt_path} -> {e}")
    return None

def collect_top_docking(docking_dfs: List[pd.DataFrame], cid: str, out_dir: str, top_n: int = 6) -> Tuple[List[str], pd.DataFrame]:
    merged = pd.concat(docking_dfs, ignore_index=True)
    sorted_df = merged.sort_values("affinity")
    ligand_states: List[str] = []
    for i in range(min(top_n, len(sorted_df))):
        try:
            row = sorted_df.iloc[i]
            packing_num = int(row["num"]); vina_mode = int(row["mode"])
            pose = f"{out_dir}/{cid}_loop_0_{packing_num}_docking_ligand_out_{vina_mode}.pdbqt"
            ligand_states.append(pose)
        except Exception as e:
            print(f"[{cid}] WARN: pick top pose failed at rank {i+1}: {e}")
    return ligand_states, sorted_df

def stage2_core_one(e: Entry, idx: int, protein_pdbqt: str, ligand_pdbqt: str, seed: str, vina_path: Optional[str]) -> Tuple[str, int, List[str], Optional[str]]:
    cid = e["complex_id"]; out_dir = e["out_dir"]; ori_lig = e["ligand_path"]
    try:
        if PACKDOCK_DIR and os.path.isdir(PACKDOCK_DIR):
            os.chdir(PACKDOCK_DIR)
        if not Path(protein_pdbqt).exists():
            raise FileNotFoundError(f"protein_pdbqt not found: {protein_pdbqt}")
        if not Path(ligand_pdbqt).exists():
            raise FileNotFoundError(f"ligand_pdbqt not found: {ligand_pdbqt}")
        ligand_states, _ = docking_with_vina(protein_pdbqt, ligand_pdbqt, ori_lig, out_dir, 0, cid, seed, idx, vina_path)
        return (cid, idx, ligand_states, None)
    except Exception as ex:
        import traceback; traceback.print_exc()
        return (cid, idx, [], f"{ex}")

def stage2_preprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    for e in chunk:
        out_dir = e["out_dir"]; cid = e["complex_id"]
        prepared = {}
        packed = e.get("stage1_packed_proteins", []) or ManifestManager.load(out_dir, cid, "stage1", "core").get("packed_proteins", [])
        for idx, packing_path in enumerate(packed):
            lig_copy = copy_file(e["ligand_path"], out_dir)
            try:
                protein_pdbqt, ligand_pdbqt = pdb2pdbqt(os.path.abspath(packing_path), lig_copy, remove_protein=False, remove_ligand=True)
                prepared[int(idx)] = {"packing_path": packing_path, "protein_pdbqt": protein_pdbqt, "ligand_pdbqt": ligand_pdbqt}
            except Exception as ex:
                print(f"[{cid}] WARN (stage2 preprocess): {_short(ex)}")
        e["_stage2_prepared"] = prepared
        ManifestManager.save(out_dir, cid, "stage2", "preprocess", {str(k): v for k, v in prepared.items()})
    return chunk

def stage2_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    try:
        if PACKDOCK_DIR and os.path.isdir(PACKDOCK_DIR):
            os.chdir(PACKDOCK_DIR)
    except Exception:
        pass
    max_workers = max(1, int(getattr(args, "docking_max_workers", 1)))
    max_inflight = getattr(args, "docking_max_inflight", None) or max_workers
    from threading import Semaphore
    sem = Semaphore(max_inflight)
    seed = getattr(args, "seed", "666")
    per_entry_states: Dict[str, Dict[int, List[str]]] = {e["complex_id"]: {} for e in chunk}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
        futs = []
        for e in chunk:
            cid = e["complex_id"]; out_dir = e["out_dir"]
            prep = e.get("_stage2_prepared", {})
            if not prep:
                loaded = ManifestManager.load(out_dir, cid, "stage2", "preprocess")
                prep = {int(k): v for k, v in loaded.items()} if loaded else {}
                if prep: print(f"[{cid}] loaded stage2 preprocess manifest ({len(prep)} items).")
            for idx, meta in sorted(prep.items(), key=lambda kv: int(kv[0])):
                sem.acquire()
                fut = ex.submit(stage2_core_one, e, int(idx), meta["protein_pdbqt"], meta["ligand_pdbqt"], seed, getattr(args, "vina_path", None))
                fut.add_done_callback(lambda _f: sem.release()); futs.append(fut)
        for fut in as_completed(futs):
            try:
                cid, idx, states, err = fut.result()
                if err: print(f"[{cid}] WARN (stage2 core): {_short(err)}")
                if states: per_entry_states[cid][idx] = states
            except Exception as ex:
                print(f"[stage2 core pool] ERROR: {ex}")
    for e in chunk:
        cid = e["complex_id"]; out_dir = e["out_dir"]
        ManifestManager.save(out_dir, cid, "stage2", "core", {"poses_by_idx": {str(k): v for k, v in per_entry_states.get(cid, {}).items()}})
        e["_stage2_core_states"] = per_entry_states.get(e["complex_id"], {})
    return chunk

def stage2_postprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    for e in chunk:
        cid = e["complex_id"]; out_dir = e["out_dir"]; ori_lig = e["ligand_path"]
        per_job_states = e.get("_stage2_core_states") or ManifestManager.load(out_dir, cid, "stage2", "core").get("poses_by_idx", {})
        per_job_states = {int(k): v for k, v in per_job_states.items()} if per_job_states else {}
        docking_dfs: List[pd.DataFrame] = []
        for idx, states in sorted(per_job_states.items()):
            log_path = f"{out_dir}/{cid}_0_{idx}_docking_ligand_vina_score_out.log"
            df = read_docking_log(log_path)
            if df is None or df.empty: continue
            if getattr(args, "rmsd", False):
                ligand_pdbqt = ori_lig
                rmsds, nums = [], []
                for pose_k, lpath in enumerate(states):
                    rmsd_txt = f"{out_dir}/rmsd_loop0_idx{idx}_pose{pose_k}.txt"
                    os.system(f"obrms {ligand_pdbqt} {lpath} > {rmsd_txt}")
                    r = read_rmsd_from_txt(rmsd_txt)
                    if r is not None:
                        rmsds.append(r); nums.append(idx)
                if rmsds:
                    df = df.head(len(rmsds)).copy(); df["true_rmsd"] = rmsds; df["num"] = nums
            else:
                df = df.copy(); df["num"] = idx
            docking_dfs.append(df)
        if docking_dfs:
            top_states, sorted_df = collect_top_docking(docking_dfs, cid, out_dir, top_n=5)
            sorted_df.to_csv(f"{out_dir}/0_dockingscores.csv", index=False)
            e["stage2_top_ligand_states"] = top_states
            ManifestManager.save(out_dir, cid, "stage2", "postprocess", {"dockingscores_csv": f"{out_dir}/0_dockingscores.csv", "top_states": top_states})
        else:
            e["stage2_top_ligand_states"] = []
            ManifestManager.save(out_dir, cid, "stage2", "postprocess", {"dockingscores_csv": None, "top_states": []})
        if "_stage2_core_states" in e: del e["_stage2_core_states"]
        if "_stage2_prepared" in e: del e["_stage2_prepared"]
    return chunk

# ---------------------------
# Stage 3 (packing)
# ---------------------------
def stage3_preprocess_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    complex_model, complex_device, complex_t2s, cmp_args = load_complex_model(args)
    side_schedule = get_t_schedule(inference_steps=args.inference_steps)
    for e in chunk:
        ManifestManager.save(e["out_dir"], e["complex_id"], "stage3", "preprocess", {"model_dir": args.model_dir, "ckpt": args.ckpt})
    group_size = max(1, int(getattr(args, "pack_group_size", 12)))
    for gstart in range(0, len(chunk), group_size):
        group = chunk[gstart:gstart + group_size]
        owners: List[dict] = []; graphs: List[HeteroData] = []
        if getattr(args, "graph_build_workers", 0) > 0 and len(group) > 1:
            with ThreadPoolExecutor(max_workers=args.graph_build_workers) as ex:
                futs = [ex.submit(_build_complex_graphs_single, e, args) for e in group]
                for f in as_completed(futs):
                    cid, gs = f.result()
                    if gs:
                        e = next(e for e in group if e["complex_id"] == cid)
                        owners.extend([e] * len(gs)); graphs.extend(gs)
        else:
            for e in group:
                cid, gs = _build_complex_graphs_single(e, args)
                if gs:
                    owners.extend([e] * len(gs)); graphs.extend(gs)
        if not graphs:
            for e in group:
                if not e.get("stage2_top_ligand_states", []):
                    e["stage3_packed_complex"] = []
            continue
        written = _pack_batch_and_write(owners=owners, orig_graphs=graphs, copies_per_graph=int(getattr(args, "stage3_packing_num", 5)),
                                       model=complex_model, device=complex_device,
                                       side_schedule=side_schedule, t_to_sigma=complex_t2s,
                                       model_args=cmp_args, args=args)
        by_cid: Dict[str, List[str]] = {}
        for p in written:
            cid = Path(p).parent.name; by_cid.setdefault(cid, []).append(p)
        for e in group:
            e["stage3_packed_complex"] = by_cid.get(e["complex_id"], [])
            old = ManifestManager.load(e["out_dir"], e["complex_id"], "stage3", "core")
            cur = old.get("packed_complex", []); cur.extend(e["stage3_packed_complex"])
            ManifestManager.save(e["out_dir"], e["complex_id"], "stage3", "core", {"packed_complex": cur})
        _del_and_gc(graphs, owners, written)
    _del_and_gc(complex_model, complex_device, complex_t2s, cmp_args, side_schedule)
    return chunk

def stage3_postprocess(entries: List[Entry], args: Namespace) -> List[Entry]:
    for e in entries:
        if not e.get("stage3_packed_complex"):
            core = ManifestManager.load(e["out_dir"], e["complex_id"], "stage3", "core")
            e["stage3_packed_complex"] = core.get("packed_complex", [])
        ManifestManager.save(e["out_dir"], e["complex_id"], "stage3", "postprocess", {
            "packed_complex": e["stage3_packed_complex"]
        })
    return entries

# ---------------------------
# Stage 4
# ---------------------------
def stage4_core_one(e: Entry, idx: int, protein_pdbqt: str, ligand_pdbqt: str, seed: str, vina_path: Optional[str]) -> Tuple[str, int, List[str], Optional[str]]:
    cid = e["complex_id"]; out_dir = e["out_dir"]; ori_lig = e["ligand_path"]
    try:
        if PACKDOCK_DIR and os.path.isdir(PACKDOCK_DIR):
            os.chdir(PACKDOCK_DIR)
        if not Path(protein_pdbqt).exists():
            raise FileNotFoundError(f"protein_pdbqt not found: {protein_pdbqt}")
        if not Path(ligand_pdbqt).exists():
            raise FileNotFoundError(f"ligand_pdbqt not found: {ligand_pdbqt}")
        ligand_states, _ = docking_with_vina(protein_pdbqt, ligand_pdbqt, ori_lig, out_dir, 1, cid, seed, idx, vina_path)
        return (cid, idx, ligand_states, None)
    except Exception as ex:
        import traceback; traceback.print_exc()
        return (cid, idx, [], f"{ex}")

def stage4_preprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    for e in chunk:
        out_dir = e["out_dir"]; cid = e["complex_id"]
        prepared = {}
        packed = e.get("stage3_packed_complex", []) or ManifestManager.load(out_dir, cid, "stage3", "core").get("packed_complex", [])
        for idx, packing_path in enumerate(packed):
            lig_copy = copy_file(e["ligand_path"], out_dir)
            try:
                protein_pdbqt, ligand_pdbqt = pdb2pdbqt(os.path.abspath(packing_path), lig_copy, remove_protein=False, remove_ligand=True)
                prepared[int(idx)] = {"packing_path": packing_path, "protein_pdbqt": protein_pdbqt, "ligand_pdbqt": ligand_pdbqt}
            except Exception as ex:
                print(f"[{cid}] WARN (stage4 preprocess): {ex}")
        e["_stage4_prepared"] = prepared
        ManifestManager.save(out_dir, cid, "stage4", "preprocess", {str(k): v for k, v in prepared.items()})
    return chunk

def stage4_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    try:
        if PACKDOCK_DIR and os.path.isdir(PACKDOCK_DIR):
            os.chdir(PACKDOCK_DIR)
    except Exception:
        pass
    max_workers = max(1, int(getattr(args, "docking_max_workers", 1)))
    max_inflight = getattr(args, "docking_max_inflight", None) or max_workers
    from threading import Semaphore
    sem = Semaphore(max_inflight)
    seed = getattr(args, "seed", "666")
    per_entry_states: Dict[str, Dict[int, List[str]]] = {e["complex_id"]: {} for e in chunk}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
        futs = []
        for e in chunk:
            cid = e["complex_id"]; out_dir = e["out_dir"]
            prep = e.get("_stage4_prepared", {})
            if not prep:
                loaded = ManifestManager.load(out_dir, cid, "stage4", "preprocess")
                prep = {int(k): v for k, v in loaded.items()} if loaded else {}
                if prep: print(f"[{cid}] loaded stage4 preprocess manifest ({len(prep)} items).")
            for idx, meta in sorted(prep.items(), key=lambda kv: int(kv[0])):
                sem.acquire()
                fut = ex.submit(stage4_core_one, e, int(idx), meta["protein_pdbqt"], meta["ligand_pdbqt"], seed, getattr(args, "vina_path", None))
                fut.add_done_callback(lambda _f: sem.release()); futs.append(fut)
        for fut in as_completed(futs):
            try:
                cid, idx, states, err = fut.result()
                if err: print(f"[{cid}] WARN (stage4 core): {_short(err)}")
                if states: per_entry_states[cid][idx] = states
            except Exception as ex:
                print(f"[stage4 core pool] ERROR: {ex}")
    for e in chunk:
        cid = e["complex_id"]; out_dir = e["out_dir"]; poses = per_entry_states.get(cid, {})
        ManifestManager.save(out_dir, cid, "stage4", "core", {"poses_by_idx": {str(k): v for k, v in poses.items()}})
        e["_stage4_core_states"] = poses
    return chunk

def stage4_postprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    for e in chunk:
        cid = e["complex_id"]; out_dir = e["out_dir"]; ori_lig = e["ligand_path"]
        per_job_states = e.get("_stage4_core_states")
        if not per_job_states:
            loaded = ManifestManager.load(out_dir, cid, "stage4", "core")
            per_job_states = loaded.get("poses_by_idx", {}) if loaded else {}
        per_job_states = {int(k): v for k, v in per_job_states.items()} if per_job_states else {}
        docking_dfs: List[pd.DataFrame] = []
        for idx, states in sorted(per_job_states.items()):
            log_path = f"{out_dir}/{cid}_1_{idx}_docking_ligand_vina_score_out.log"
            df = read_docking_log(log_path)
            if df is None or df.empty: continue
            if getattr(args, "rmsd", False):
                rmsds, nums = [], []
                for pose_k, lpath in enumerate(states):
                    try:
                        rmsd_txt = f"{out_dir}/rmsd_loop1_idx{idx}_pose{pose_k}.txt"
                        os.system(f"obrms {ori_lig} {lpath} > {rmsd_txt}")
                        r = read_rmsd_from_txt(rmsd_txt)
                        if r is not None:
                            rmsds.append(r); nums.append(idx)
                    except Exception as ex:
                        print(f"[{cid}] WARN: RMSD fail idx={idx} pose={pose_k}: {_short(ex)}")
                if rmsds:
                    df = df.head(len(rmsds)).copy(); df["true_rmsd"] = rmsds; df["num"] = nums
                else:
                    df = df.copy(); df["num"] = idx
            else:
                df = df.copy(); df["num"] = idx
            docking_dfs.append(df)
        if docking_dfs:
            merged = pd.concat(docking_dfs, ignore_index=True)
            score_path = f"{out_dir}/1_dockingscores.csv"
            if "true_rmsd" in merged.columns:
                merged.sort_values("true_rmsd").to_csv(score_path, index=False)
            else:
                merged.sort_values("affinity").to_csv(score_path, index=False)
            ManifestManager.save(out_dir, cid, "stage4", "postprocess", {
                "dockingscores_csv": score_path,
                "poses_by_idx": {str(k): v for k, v in per_job_states.items()},
            })
        else:
            ManifestManager.save(out_dir, cid, "stage4", "postprocess", {
                "dockingscores_csv": None,
                "poses_by_idx": {str(k): v for k, v in per_job_states.items()},
            })
        if "_stage4_core_states" in e: del e["_stage4_core_states"]
    return chunk


# ---------------------------
# Orchestration
# ---------------------------
def main():
    args = parse_arguments()

    print("▶ Stage 0: prepare & pocket extraction ...")
    entries = stage0_prepare_and_pocket(args)
    print(f"Prepared {len(entries)} entries.")

    print("▶ Stage 1 preprocess+core: ligand-free protein packing ...")
    entries = run_per_gpu(entries, stage1_preprocess_core_worker, args)
    print("▶ Stage 1 postprocess.")
    entries = stage1_postprocess(entries, args)

    print("▶ Stage 2 preprocess: pdb2pdbqt ...")
    entries = run_per_gpu(entries, stage2_preprocess_worker, args)
    print("▶ Stage 2 core: docking on apo packed conformations ...")
    entries = run_per_gpu(entries, stage2_core_worker, args)
    print("▶ Stage 2 postprocess: parse logs & pick top ...")
    entries = run_per_gpu(entries, stage2_postprocess_worker, args)

    print("▶ Stage 3 preprocess+core: ligand-conditioned complex packing (with Stage2 top poses) ...")
    entries = run_per_gpu(entries, stage3_preprocess_core_worker, args)
    print("▶ Stage 3 postprocess.")
    entries = stage3_postprocess(entries, args)

    print("▶ Stage 4 preprocess: pdb2pdbqt ...")
    entries = run_per_gpu(entries, stage4_preprocess_worker, args)
    print("▶ Stage 4 core: final docking on packed complexes ...")
    entries = run_per_gpu(entries, stage4_core_worker, args)
    print("▶ Stage 4 postprocess: parse logs & write scores ...")
    entries = run_per_gpu(entries, stage4_postprocess_worker, args)

    print("✅ All entries processed.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
