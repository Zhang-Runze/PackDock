#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

PackDock docking pipeline.

Stages:
0. prepare & pocket extraction
1. protein ligand-free side-chain packing 
2. docking on ligand-free packed structures
3. protein ligand-conditioned side-chain packing 
4. docking on ligand-conditioned packed structures

"""

from __future__ import annotations

import copy
import gc
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
import warnings
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from Bio.PDB import PDBIO, PDBParser, Select
from rdkit import Chem, RDLogger
from torch_geometric.data import HeteroData

# Local project imports
from datasets.pdbbind import PDBBind, read_mol, read_pdb
from datasets.process_mols import (
    extract_receptor_structure,
    get_lig_graph_with_matching,
    get_rec_graph,
    parse_pdb_from_path,
    read_molecule,
)
from utils.diffusion_utils import get_t_schedule
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.sampling import randomize_position, sampling
from utils.utils import (
    extract_protein_packing_state,
    get_ligand_outside_num,
    get_model,
    gpu_docking_with_ligand_boxsize_sdf,
    pdb2pdbqt,
    read_first_docking_log,
)
from utils.visualise import PDBFile

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.DisableLog("rdApp.*")


PACKDOCK_HOME = os.environ.get("PACKDOCK_HOME", os.getcwd())

Entry = Dict[str, Any]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Protein-ligand docking evaluation with side-chain packing",
        formatter_class=RawDescriptionHelpFormatter,
    )

    group = parser.add_argument_group("Model configuration")
    group.add_argument("--protein_model_dir", type=str, default="workdir/protein_model")
    group.add_argument("--model_dir", type=str, default="workdir/complex_model")
    group.add_argument("--ckpt", type=str, default="best_ema_inference_epoch_model.pt")

    group = parser.add_argument_group("Sampling parameters")
    group.add_argument("--inference_steps", type=int, default=20)
    group.add_argument("--actual_steps", type=int, default=20)
    group.add_argument("--no_random", action="store_true", default=False)
    group.add_argument("--ode", action="store_true", default=False)
    group.add_argument("--batch_size", type=int, default=36)
    group.add_argument("--no_final_step_noise", action="store_true", default=False)

    group = parser.add_argument_group("Speed and batching")
    group.add_argument("--pack_group_size", type=int, default=1)
    group.add_argument("--graph_build_workers", type=int, default=0)

    group = parser.add_argument_group("Docking parallelism")
    group.add_argument("--docking_max_workers", type=int, default=1)
    group.add_argument("--docking_max_inflight", type=int, default=None)

    group = parser.add_argument_group("OpenMM side-chain relax")
    group.add_argument(
        "--relax-stage1",
        dest="relax_stage1",
        action="store_true",
        help="Run OpenMM relax after stage 1 (default: enabled)",
        default=True,
    )
    group.add_argument("--no-relax-stage1", dest="relax_stage1", action="store_false")
    group.add_argument(
        "--relax-stage3",
        dest="relax_stage3",
        action="store_true",
        help="Run OpenMM relax after stage 3 (default: enabled)",
        default=True,
    )
    group.add_argument("--no-relax-stage3", dest="relax_stage3", action="store_false")
    group.add_argument("--openmm-platform", type=str, default="CUDA")
    group.add_argument("--openmm-md-steps", type=int, default=0)
    group.add_argument("--openmm-explicit", action="store_true", default=False)
    group.add_argument("--openmm-ph", type=float, default=7.0)
    group.add_argument("--openmm-k-restraint", type=float, default=1000.0)

    group = parser.add_argument_group("Receptor parameters")
    group.add_argument("--receptor_radius", type=float, default=15)
    group.add_argument("--c_alpha_max_neighbors", type=int, default=24)
    group.add_argument("--all_atoms", action="store_true", default=True)
    group.add_argument("--atom_radius", type=float, default=5)
    group.add_argument("--atom_max_neighbors", type=int, default=8)
    group.add_argument("--remove_hs", action="store_true", default=True)

    group = parser.add_argument_group("Matching parameters")
    group.add_argument("--matching_popsize", type=int, default=20)
    group.add_argument("--matching_maxiter", type=int, default=20)
    group.add_argument("--num_conformers", type=int, default=1)

    group = parser.add_argument_group("I/O")
    group.add_argument("--input_csv", type=str, required=True)
    group.add_argument("--out_dir", type=str, default="results/test")
    group.add_argument("--seed", type=str, default="666")
    group.add_argument("--save_visualisation", action="store_true", default=True)
    group.add_argument("--rmsd", action="store_true", default=False)

    group = parser.add_argument_group("Pocket extraction")
    group.add_argument("--extract_pocket", action="store_true", default=True)
    group.add_argument("--pocket_radius", type=float, default=5.0)
    group.add_argument("--include_backbone", action="store_true", default=True)
    group.add_argument("--min_pocket_radius", type=int, default=3)

    group = parser.add_argument_group("Packing parameters")
    group.add_argument(
        "--stage1_packing_num",
        type=int,
        default=1,
        help="Number of packing conformations per protein graph in stage 1",
    )
    group.add_argument(
        "--stage3_packing_num",
        type=int,
        default=5,
        help="Number of packing conformations per complex graph in stage 3",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def copy_file(src: str, dst_dir: str) -> str:
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(dst_dir) / Path(src).name
    shutil.copy(src, out_path)
    return str(out_path)


def run_obabel_inplace_pdb(pdb_path: str) -> bool:
    if not shutil.which("obabel"):
        print(f"[normalize] WARNING: obabel not found in PATH; skipping: {pdb_path}")
        return False
    try:
        subprocess.run(
            ["obabel", "-ipdb", pdb_path, "-O", pdb_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[normalize] WARNING: obabel failed on {pdb_path}: {exc}")
        return False


def rdkit_can_read_pdb(pdb_path: str) -> bool:
    try:
        mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=True)
        return mol is not None
    except Exception:
        return False


def get_packing_num(state_path: str) -> int:

    stem = Path(state_path).stem

    patterns = [
        r"(?:^|[_-])sample-(\d+)(?!.*(?:^|[_-])sample-\d+)",
        r"(?:^|[_-])model[_-]?(\d+)(?!.*(?:^|[_-])model[_-]?\d+)",
        r"(?:^|[_-])state[_-]?(\d+)(?!.*(?:^|[_-])state[_-]?\d+)",
        r"(?:^|[_-])pack(?:ing)?[_-]?(\d+)(?!.*(?:^|[_-])pack(?:ing)?[_-]?\d+)",
        r"(?:^|[_-])seed-(\d+)(?!.*(?:^|[_-])seed-\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1)) + 1

    numbers = re.findall(r"(\d+)", stem)
    if numbers:
        return int(numbers[-1]) + 1

    raise ValueError(f"Unable to parse packing index from: {stem}")


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------
def manifest_path(out_dir: str, cid: str, stage: str, part: str) -> Path:
    return Path(out_dir) / f"{cid}__{stage}__{part}.json"


def save_manifest(out_dir: str, cid: str, stage: str, part: str, payload: Dict[str, Any]) -> None:
    path = manifest_path(out_dir, cid, stage, part)
    try:
        path.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"[{cid}] WARNING: failed to write manifest {path}: {exc}")


def load_manifest(out_dir: str, cid: str, stage: str, part: str) -> Dict[str, Any]:
    path = manifest_path(out_dir, cid, stage, part)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[{cid}] WARNING: failed to read manifest {path}: {exc}")
        return {}


# ---------------------------------------------------------------------
# Pocket extraction (stage 0)
# ---------------------------------------------------------------------
def ligand_coords_from_any(ligand_path: str) -> Optional[np.ndarray]:
    try:
        lower_path = ligand_path.lower()
        if lower_path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(ligand_path, sanitize=False, removeHs=False)
        elif lower_path.endswith(".sdf"):
            mol = Chem.SDMolSupplier(ligand_path, sanitize=False, removeHs=False)[0]
        elif lower_path.endswith(".mol2"):
            mol = Chem.MolFromMol2File(ligand_path, sanitize=False, removeHs=False)
        elif lower_path.endswith(".mol"):
            mol = Chem.MolFromMolFile(ligand_path, sanitize=False, removeHs=False)
        else:
            mol = Chem.MolFromPDBFile(ligand_path, sanitize=False, removeHs=False)

        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            return np.array(
                [
                    [
                        conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z,
                    ]
                    for i in range(mol.GetNumAtoms())
                ]
            )
    except Exception as exc:
        print(f"[ligand] RDKit parsing failed for {ligand_path}: {exc}")

    try:
        if ligand_path.lower().endswith(".pdb"):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("ligand", ligand_path)
            coords = [atom.get_coord() for atom in structure.get_atoms()]
            if coords:
                return np.array(coords)
    except Exception as exc:
        print(f"[ligand] BioPython parsing failed for {ligand_path}: {exc}")

    return None


def candidate_radii(
    base: int,
    deltas: Tuple[int, ...] = (0, -1, +1, -2, +2, -3, +3),
    min_radius: int = 2,
) -> List[int]:
    seen = set()
    radii: List[int] = []
    for delta in deltas:
        radius = base + delta
        if radius >= min_radius and radius not in seen:
            radii.append(radius)
            seen.add(radius)
    return radii


def extract_protein_pocket(
    protein_path: str,
    ligand_path: str,
    pocket_radius: float = 5.0,
    output_path: Optional[str] = None,
    include_backbone: bool = True,
) -> str:
    if output_path is None:
        protein_stem = Path(protein_path).with_suffix("")
        ligand_stem = Path(ligand_path).stem
        output_path = str(
            Path(protein_path).parent / f"{protein_stem.name}_{ligand_stem}_pocket_{int(pocket_radius)}A.pdb"
        )

    try:
        parser = PDBParser(QUIET=True)
        protein_structure = parser.get_structure("protein", protein_path)
        ligand_coords = ligand_coords_from_any(ligand_path)
        if ligand_coords is None:
            print(f"[pocket] WARNING: no ligand coordinates found; using original receptor: {protein_path}")
            return protein_path
    except Exception as exc:
        print(f"[pocket] ERROR: failed to parse inputs: {exc}; using original receptor")
        return protein_path

    class PocketSelect(Select):
        def accept_residue(self, residue) -> bool:
            residue_coords = []
            for atom in residue:
                if include_backbone or atom.get_name() not in ["N", "CA", "C", "O"]:
                    residue_coords.append(atom.get_coord())
            if not residue_coords:
                return False
            residue_coords_array = np.asarray(residue_coords, dtype=float)
            distances = np.linalg.norm(
                residue_coords_array[:, None, :] - ligand_coords[None, :, :],
                axis=2,
            )
            return float(np.min(distances)) <= float(pocket_radius)

    try:
        io = PDBIO()
        io.set_structure(protein_structure)
        io.save(output_path, PocketSelect())
        if Path(output_path).exists() and "ATOM" in Path(output_path).read_text():
            run_obabel_inplace_pdb(output_path)
            if rdkit_can_read_pdb(output_path):
                return output_path
    except Exception as exc:
        print(f"[pocket] WARNING: failed to write or validate pocket at {pocket_radius} Å: {exc}")

    return protein_path


def try_extract_pocket_robust(entry: Entry, args: Namespace) -> Entry:
    cid = entry["complex_id"]
    protein_path = entry["protein_path"]
    ligand_path = entry["ligand_path"]
    out_dir = entry["out_dir"]

    if not Path(protein_path).exists():
        print(f"[{cid}] WARNING: protein not found: {protein_path}")
        entry["working_protein"] = protein_path
        return entry
    if not Path(ligand_path).exists():
        print(f"[{cid}] WARNING: ligand not found: {ligand_path}")
        entry["working_protein"] = protein_path
        return entry

    if not getattr(args, "extract_pocket", False):
        entry["working_protein"] = protein_path
        save_manifest(
            out_dir,
            cid,
            "stage0",
            "postprocess",
            {
                "working_protein": protein_path,
                "pocket_radius_used": None,
                "pocket_attempts_used": 0,
                "protein_path": protein_path,
                "ligand_path": ligand_path,
            },
        )
        return entry

    base_radius = int(getattr(args, "pocket_radius", 7))
    min_radius = int(getattr(args, "min_pocket_radius", 3))
    include_backbone = bool(getattr(args, "include_backbone", False))
    radii = candidate_radii(base_radius, (0, -1, +1, -2, +2, -3, +3), min_radius)

    for attempt_idx, radius in enumerate(radii, start=1):
        output_path = str(Path(out_dir) / f"{cid}_pocket_{radius}A.pdb")
        try:
            pocket_path = extract_protein_pocket(
                protein_path,
                ligand_path,
                radius,
                output_path,
                include_backbone,
            )
            if rdkit_can_read_pdb(pocket_path):
                entry.update(
                    {
                        "working_protein": pocket_path,
                        "pocket_radius_used": radius,
                        "pocket_attempts_used": attempt_idx,
                    }
                )
                save_manifest(
                    out_dir,
                    cid,
                    "stage0",
                    "postprocess",
                    {
                        "working_protein": pocket_path,
                        "pocket_radius_used": radius,
                        "pocket_attempts_used": attempt_idx,
                        "protein_path": protein_path,
                        "ligand_path": ligand_path,
                    },
                )
                return entry
        except Exception as exc:
            print(f"[{cid}] WARNING: pocket extraction at {radius} Å failed: {exc}")

    print(f"[{cid}] WARNING: all pocket extraction attempts failed; using original receptor")
    entry.update(
        {
            "working_protein": protein_path,
            "pocket_radius_used": None,
            "pocket_attempts_used": len(radii),
        }
    )
    save_manifest(
        out_dir,
        cid,
        "stage0",
        "postprocess",
        {
            "working_protein": protein_path,
            "pocket_radius_used": None,
            "pocket_attempts_used": len(radii),
            "protein_path": protein_path,
            "ligand_path": ligand_path,
        },
    )
    return entry


# ---------------------------------------------------------------------
# Restore full protein from pocket and optionally run OpenMM relax
# ---------------------------------------------------------------------
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def restore_full_protein_from_pocket(
    full_pdb: str,
    pocket_pdb: str,
    out_path: str,
    replace_backbone: bool = False,
) -> str:
    parser = PDBParser(QUIET=True)
    full_structure = parser.get_structure("full", full_pdb)
    pocket_structure = parser.get_structure("pocket", pocket_pdb)

    pocket_map = {}
    for residue in pocket_structure.get_residues():
        chain_id = residue.get_parent().id
        het_flag, seq_id, insertion_code = residue.id
        key = (chain_id, het_flag, seq_id, insertion_code)
        pocket_map[key] = {atom.name: atom for atom in residue.get_atoms()}

    for residue in full_structure.get_residues():
        chain_id = residue.get_parent().id
        het_flag, seq_id, insertion_code = residue.id
        key = (chain_id, het_flag, seq_id, insertion_code)
        if key not in pocket_map:
            continue
        pocket_atoms = pocket_map[key]
        for atom in residue.get_atoms():
            if (not replace_backbone) and atom.name in BACKBONE_ATOMS:
                continue
            if atom.name in pocket_atoms:
                atom.set_coord(pocket_atoms[atom.name].get_coord())

    io = PDBIO()
    io.set_structure(full_structure)
    io.save(out_path)
    return out_path


def _load_and_prepare_compat(pdb_path: str, ph: float = 7.0, keep_het: bool = False, remove_waters: bool = True):
    from pdbfixer import PDBFixer
    from openmm import app

    fixer = PDBFixer(filename=str(pdb_path))
    if not keep_het:
        fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph)

    modeller = app.Modeller(fixer.topology, fixer.positions)
    if keep_het and remove_waters:
        water_names = {"HOH", "WAT", "DOD", "HOD", "TIP3", "TIP", "SOL"}
        to_delete = []
        for residue in modeller.topology.residues():
            if residue.name.upper() in water_names:
                to_delete.extend(list(residue.atoms()))
        if to_delete:
            modeller.delete(to_delete)
    return modeller


def _add_backbone_restraints(system, topology, ref_positions, k_kj_per_mol_nm2: float = 1000.0):
    import openmm as mm
    from openmm import unit

    force = mm.CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addPerParticleParameter("k")

    backbone_atom_names = {"N", "CA", "C", "O"}
    for idx, atom in enumerate(topology.atoms()):
        if atom.name in backbone_atom_names and atom.element is not None:
            x, y, z = ref_positions[idx].value_in_unit(unit.nanometer)
            force.addParticle(idx, [x, y, z, k_kj_per_mol_nm2])

    system.addForce(force)


def openmm_sidechain_relax(
    pdb_in: str,
    pdb_out: str,
    ph: float = 7.0,
    explicit: bool = False,
    md_steps: int = 0,
    platform: Optional[str] = None,
    k_restraint: float = 1000.0,
) -> str:
    try:
        import openmm as mm
        from openmm import Platform, app, unit
        from openmm import LocalEnergyMinimizer
    except Exception as exc:
        print(f"[openmm] WARNING: OpenMM unavailable ({exc}); copying input to output")
        shutil.copyfile(pdb_in, pdb_out)
        return pdb_out

    modeller = _load_and_prepare_compat(pdb_in, ph=ph, keep_het=False, remove_waters=True)
    if explicit:
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=1.0 * unit.nanometer,
            ionicStrength=0.15 * unit.molar,
        )
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )
    else:
        forcefield = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )

    _add_backbone_restraints(system, modeller.topology, modeller.positions, k_kj_per_mol_nm2=k_restraint)
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds,
    )

    if platform:
        selected_platform = Platform.getPlatformByName(platform)
        simulation = app.Simulation(modeller.topology, system, integrator, selected_platform)
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)
    LocalEnergyMinimizer.minimize(simulation.context, tolerance=1.0, maxIterations=2000)

    if md_steps and md_steps > 0:
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        simulation.step(int(md_steps))

    state = simulation.context.getState(getPositions=True)
    with open(pdb_out, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), handle)
    return pdb_out


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------
def construct_protein_graph(protein_path: str, pdb_id: str, ligand: Any, args: Namespace) -> List[HeteroData]:
    protein_graphs: List[HeteroData] = []
    rec_model, pdb_mol = parse_pdb_from_path(protein_path)
    protein_graph = HeteroData()
    protein_graph["name"] = str(pdb_id)

    (
        rec,
        rec_coords,
        c_alpha_coords,
        rotamer_idx,
        backbone_idx,
        physchem,
        psp19,
        n_coords,
        c_coords,
        lm_embeddings,
    ) = extract_receptor_structure(copy.deepcopy(rec_model), ligand)

    cutoff = 3.5
    pocket_outside_num = get_ligand_outside_num(rec_coords, ligand, cutoff)
    pocket_outside_idx = [
        np.ones_like(arr) if i in pocket_outside_num else arr
        for i, arr in enumerate(backbone_idx)
    ]

    get_rec_graph(
        rec,
        pdb_mol,
        rec_coords,
        c_alpha_coords,
        rotamer_idx,
        backbone_idx,
        physchem,
        psp19,
        n_coords,
        c_coords,
        pocket_outside_num,
        pocket_outside_idx,
        protein_graph,
        rec_radius=args.receptor_radius,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        all_atoms=args.all_atoms,
        atom_radius=args.atom_radius,
        atom_max_neighbors=args.atom_max_neighbors,
        remove_hs=args.remove_hs,
        lm_embeddings=lm_embeddings,
    )

    center = torch.mean(protein_graph["receptor"].pos, dim=0, keepdim=True)
    protein_graph["receptor"].pos -= center
    protein_graph["atom"].pos -= center
    protein_graph.original_center = center
    protein_graphs.append(protein_graph)
    return protein_graphs


def _read_ligand_state_any(pdbqt_path: str) -> Optional[Any]:
    for ext in (".pdbqt", ".mol", ".sdf", ".mol2"):
        try:
            candidate_path = pdbqt_path if ext == ".pdbqt" else pdbqt_path.replace(".pdbqt", ext)
            if ext != ".pdbqt":
                os.system(f"obabel -ipdbqt {pdbqt_path} -O {candidate_path}")
            ligand = read_molecule(candidate_path, remove_hs=False, sanitize=False)
            if ext != ".pdbqt" and Path(candidate_path).exists():
                Path(candidate_path).unlink(missing_ok=True)
            if ligand is not None:
                return ligand
        except Exception:
            continue
    return None


def construct_complex_graphs(
    ligand_states: List[str],
    ref_lig: Any,
    protein_path: str,
    out_dir: str,
    pdb_id: str,
    args: Namespace,
    loop_num: int,
) -> Tuple[List[HeteroData], str]:
    graphs: List[HeteroData] = []
    pdb_path_used = ""

    for idx, state_path in enumerate(ligand_states):
        try:
            packing_num = get_packing_num(state_path)
            pdb_path_used = f"{out_dir}/packing_{pdb_id}_{pdb_id}_{packing_num}new.pdb"
            ligand = _read_ligand_state_any(state_path)
            if ligand is None:
                print(f"[{pdb_id}] WARNING: unable to read ligand state: {state_path}")
                continue

            rec_model, pdb_mol = parse_pdb_from_path(protein_path)
            data = HeteroData()
            data["name"] = f"{loop_num}_{idx + 1}"

            (
                rec,
                rec_coords,
                c_alpha_coords,
                rotamer_idx,
                backbone_idx,
                physchem,
                psp19,
                n_coords,
                c_coords,
                lm_embeddings,
            ) = extract_receptor_structure(copy.deepcopy(rec_model), ligand)

            cutoff = 3.5
            ligand_outside_num = get_ligand_outside_num(rec_coords, ref_lig, cutoff)
            ligand_outside_idx = [
                np.ones_like(arr) if i in ligand_outside_num else arr
                for i, arr in enumerate(backbone_idx)
            ]

            get_rec_graph(
                rec,
                pdb_mol,
                rec_coords,
                c_alpha_coords,
                rotamer_idx,
                backbone_idx,
                physchem,
                psp19,
                n_coords,
                c_coords,
                ligand_outside_num,
                ligand_outside_idx,
                data,
                rec_radius=args.receptor_radius,
                c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                all_atoms=args.all_atoms,
                atom_radius=args.atom_radius,
                atom_max_neighbors=args.atom_max_neighbors,
                remove_hs=args.remove_hs,
                lm_embeddings=lm_embeddings,
            )
            get_lig_graph_with_matching(
                ligand,
                data,
                args.matching_popsize,
                args.matching_maxiter,
                False,
                True,
                args.num_conformers,
                remove_hs=args.remove_hs,
            )

            center = torch.mean(data["receptor"].pos, dim=0, keepdim=True)
            data["receptor"].pos -= center
            data["atom"].pos -= center
            data["ligand"].pos -= center
            data.original_center = center
            graphs.append(data)
        except Exception as exc:
            print(f"[{pdb_id}] WARNING: invalid packed conformation skipped ({pdb_path_used}): {exc}")
            continue

    return graphs, pdb_path_used


def _build_protein_graph_single(entry: Entry, args: Namespace) -> Tuple[str, List[HeteroData]]:
    cid = entry["complex_id"]
    working_protein = entry.get("working_protein", entry["protein_path"])
    ligand_path = entry["ligand_path"]

    try:
        if ligand_path.endswith(".pdb"):
            ligand = Chem.MolFromPDBFile(ligand_path, sanitize=True, removeHs=True)
        elif ligand_path.endswith(".sdf"):
            ligand = Chem.SDMolSupplier(ligand_path, sanitize=True, removeHs=True)[0]
        else:
            ligand = Chem.MolFromPDBFile(ligand_path, sanitize=True, removeHs=True)
        graphs = construct_protein_graph(working_protein, cid, ligand, args)
        return cid, graphs
    except Exception as exc:
        print(f"[{cid}] protein graph construction failed: {exc}")
        return cid, []


def _build_complex_graphs_single(entry: Entry, args: Namespace) -> Tuple[str, List[HeteroData]]:
    cid = entry["complex_id"]
    out_dir = entry["out_dir"]
    working_protein = entry.get("working_protein", entry["protein_path"])
    top_states = entry.get("stage2_top_ligand_states", [])
    ligand_path = entry["ligand_path"]

    if not top_states:
        return cid, []

    try:
        if ligand_path.endswith(".pdb"):
            ref_ligand = Chem.MolFromPDBFile(ligand_path, sanitize=True, removeHs=True)
        elif ligand_path.endswith(".sdf"):
            ref_ligand = Chem.SDMolSupplier(ligand_path, sanitize=True, removeHs=True)[0]
        else:
            raise ValueError(f"Unsupported ligand reference format: {ligand_path}")

        graphs, _ = construct_complex_graphs(
            top_states,
            ref_ligand,
            working_protein,
            out_dir,
            cid,
            args,
            loop_num=1,
        )
        return cid, graphs
    except Exception as exc:
        print(f"[{cid}] complex graph construction failed: {exc}")
        return cid, []


# ---------------------------------------------------------------------
# Shared packing core
# ---------------------------------------------------------------------
def _pack_batch_and_write(
    owners: List[Entry],
    orig_graphs: List[HeteroData],
    copies_per_graph: int,
    model: Any,
    device: torch.device,
    side_schedule: Any,
    t_to_sigma: Any,
    model_args: Namespace,
    args: Namespace,
) -> List[str]:
    data_list: List[HeteroData] = []
    owner_index: List[int] = []

    for i, graph in enumerate(orig_graphs):
        for _ in range(copies_per_graph):
            data_list.append(copy.deepcopy(graph))
            owner_index.append(i)

    if not data_list:
        return []

    randomize_position(data_list, False, False)
    visualizations = None

    if args.save_visualisation:
        visualizations = []
        for idx, sample in enumerate(data_list):
            entry = owners[owner_index[idx]]
            original_protein = entry.get("working_protein", entry["protein_path"])
            mol = Chem.MolFromPDBFile(original_protein, sanitize=True, removeHs=True)
            pdb = PDBFile(mol)
            pdb.add(mol, 0, 0)
            pdb.add((sample["atom"].pos + sample.original_center).detach().cpu(), 1, 0)
            visualizations.append(pdb)

    with torch.no_grad():
        data_list, _ = sampling(
            data_list=data_list,
            model=model,
            inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
            side_schedule=side_schedule,
            device=device,
            t_to_sigma=t_to_sigma,
            model_args=model_args,
            no_random=args.no_random,
            ode=args.ode,
            visualization_list=visualizations,
            confidence_model=None,
            confidence_data_list=None,
            confidence_model_args=None,
            batch_size=args.batch_size,
            no_final_step_noise=args.no_final_step_noise,
        )

    written_paths: List[str] = []
    local_counts: Dict[str, int] = {}
    for idx, sample in enumerate(data_list):
        entry = owners[owner_index[idx]]
        cid = entry["complex_id"]
        local_counts.setdefault(cid, 0)
        local_counts[cid] += 1

        out_dir = entry["out_dir"]
        out_pdb = f"{out_dir}/packing_{sample['name']}_{cid}_{local_counts[cid]}.pdb"

        if args.save_visualisation and visualizations is not None:
            vis = visualizations[idx]
            vis.add((sample["atom"].pos + sample.original_center).detach().cpu(), part=1, order=1)
            vis.write(out_pdb)
        else:
            original_protein = entry.get("working_protein", entry["protein_path"])
            mol = Chem.MolFromPDBFile(original_protein, sanitize=True, removeHs=True)
            pdb = PDBFile(mol)
            pdb.add(mol, 0, 0)
            pdb.add((sample["atom"].pos + sample.original_center).detach().cpu(), 1, 0)
            pdb.write(out_pdb)

        written_paths.append(out_pdb)

    _del_and_gc(data_list, visualizations)
    return written_paths


# ---------------------------------------------------------------------
# Docking helpers
# ---------------------------------------------------------------------
def _short(msg: str, maxlen: int = 180) -> str:
    text = str(msg)
    return text if len(text) <= maxlen else text[: maxlen - 3] + "..."


def read_rmsd_from_txt(txt_path: str) -> Optional[float]:
    try:
        for line in Path(txt_path).read_text().splitlines():
            if line.startswith("RMSD"):
                return float(line.strip().split()[-1])
    except Exception as exc:
        print(f"[rmsd] WARNING: failed to parse {txt_path}: {exc}")
    return None


def collect_top_docking(
    docking_dfs: List[pd.DataFrame],
    cid: str,
    out_dir: str,
    top_n: int = 6,
) -> Tuple[List[str], pd.DataFrame]:
    merged = pd.concat(docking_dfs, ignore_index=True)
    sorted_df = merged.sort_values("affinity")
    ligand_states: List[str] = []

    for i in range(min(top_n, len(sorted_df))):
        try:
            row = sorted_df.iloc[i]
            packing_num = int(row["num"])
            vina_mode = int(row["mode"])
            pose = f"{out_dir}/{cid}_loop_0_{packing_num}_docking_ligand_out_{vina_mode}.pdbqt"
            ligand_states.append(pose)
        except Exception as exc:
            print(f"[{cid}] WARNING: failed to select top pose at rank {i + 1}: {exc}")

    return ligand_states, sorted_df


def stage2_core_one(
    entry: Entry,
    idx: int,
    protein_pdbqt: str,
    ligand_pdbqt: str,
    seed: str,
) -> Tuple[str, int, List[str], Optional[str]]:
    cid = entry["complex_id"]
    out_dir = entry["out_dir"]
    original_ligand = entry["ligand_path"]

    try:
        if PACKDOCK_HOME and os.path.isdir(PACKDOCK_HOME):
            os.chdir(PACKDOCK_HOME)
        if not Path(protein_pdbqt).exists():
            raise FileNotFoundError(f"protein_pdbqt not found: {protein_pdbqt}")
        if not Path(ligand_pdbqt).exists():
            raise FileNotFoundError(f"ligand_pdbqt not found: {ligand_pdbqt}")

        ligand_states, _ = gpu_docking_with_ligand_boxsize_sdf(
            protein_pdbqt,
            ligand_pdbqt,
            original_ligand,
            out_dir,
            0,
            cid,
            seed,
            idx,
        )
        return cid, idx, ligand_states, None
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return cid, idx, [], str(exc)


def stage4_core_one(
    entry: Entry,
    idx: int,
    protein_pdbqt: str,
    ligand_pdbqt: str,
    seed: str,
) -> Tuple[str, int, List[str], Optional[str]]:
    cid = entry["complex_id"]
    out_dir = entry["out_dir"]
    original_ligand = entry["ligand_path"]

    try:
        if PACKDOCK_HOME and os.path.isdir(PACKDOCK_HOME):
            os.chdir(PACKDOCK_HOME)
        if not Path(protein_pdbqt).exists():
            raise FileNotFoundError(f"protein_pdbqt not found: {protein_pdbqt}")
        if not Path(ligand_pdbqt).exists():
            raise FileNotFoundError(f"ligand_pdbqt not found: {ligand_pdbqt}")

        ligand_states, _ = gpu_docking_with_ligand_boxsize_sdf(
            protein_pdbqt,
            ligand_pdbqt,
            original_ligand,
            out_dir,
            1,
            cid,
            seed,
            idx,
        )
        return cid, idx, ligand_states, None
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return cid, idx, [], str(exc)


# ---------------------------------------------------------------------
# Model loading and memory cleanup
# ---------------------------------------------------------------------
def load_protein_model(args: Namespace) -> Tuple[Any, torch.device, Any, Namespace]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(f"{args.protein_model_dir}/model_parameters.yml") as handle:
        model_args = Namespace(**yaml.full_load(handle))
    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    model = get_model(model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, only_protein=True)
    state = torch.load(f"{args.protein_model_dir}/{args.ckpt}", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device, t_to_sigma, model_args


def load_complex_model(args: Namespace) -> Tuple[Any, torch.device, Any, Namespace]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(f"{args.model_dir}/model_parameters.yml") as handle:
        model_args = Namespace(**yaml.full_load(handle))
    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    model = get_model(model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    state = torch.load(f"{args.model_dir}/{args.ckpt}", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device, t_to_sigma, model_args


def _cuda_gc() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _del_and_gc(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    _cuda_gc()


def has_full_results(out_dir: str, min_rows: int = 6) -> bool:
    score_path = Path(out_dir) / "1_dockingscores.csv"
    if not score_path.exists():
        return False

    try:
        df = pd.read_csv(score_path)
    except Exception:
        return False

    if len(df) < int(min_rows):
        return False
    if "true_rmsd" not in df.columns:
        return False

    rmsd = pd.to_numeric(df["true_rmsd"], errors="coerce").dropna()
    if rmsd.empty:
        return False
    if rmsd.min() > 2:
        return False
    return True


# ---------------------------------------------------------------------
# GPU scheduling helpers
# ---------------------------------------------------------------------
def available_gpu_ids() -> List[int]:
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def chunk_round_robin(items: List[Any], ngpu: int) -> List[List[Any]]:
    if ngpu <= 0:
        return [items]
    chunks = [[] for _ in range(ngpu)]
    for i, item in enumerate(items):
        chunks[i % ngpu].append(item)
    return chunks


def run_per_gpu(entries: List[Entry], worker, args: Namespace) -> List[Entry]:
    gpus = available_gpu_ids()
    if not gpus:
        return worker(entries, args, 0)

    chunks = chunk_round_robin(entries, len(gpus))
    results: List[Entry] = []
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(worker, chunk, args, gid) for gid, chunk in zip(gpus, chunks)]
        for future in as_completed(futures):
            results.extend(future.result())

    results.sort(key=lambda d: d["complex_id"])
    return results


# ---------------------------------------------------------------------
# Stage 0 helpers
# ---------------------------------------------------------------------
def read_input_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    required = ["protein_path", "ligand_path", "complex_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required)
    if df.empty:
        raise ValueError("No valid rows found in the input CSV")

    print(f"Loaded {len(df)} entries from {csv_file}")
    return df


def prepare_entries(args: Namespace) -> List[Entry]:
    df = read_input_csv(args.input_csv)
    root_out_dir = Path(args.out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Entry] = []
    for _, row in df.iterrows():
        cid = str(row["complex_id"])
        entry_out_dir = root_out_dir / cid
        entry_out_dir.mkdir(parents=True, exist_ok=True)
        entries.append(
            {
                "complex_id": cid,
                "protein_path": str(row["protein_path"]),
                "ligand_path": str(row["ligand_path"]),
                "out_dir": str(os.path.abspath(entry_out_dir)),
            }
        )
    return entries


# ---------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------
def stage1_preprocess_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    protein_model, protein_device, protein_t2s, model_args = load_protein_model(args)
    side_schedule = get_t_schedule(inference_steps=args.inference_steps)

    for entry in chunk:
        save_manifest(
            entry["out_dir"],
            entry["complex_id"],
            "stage1",
            "preprocess",
            {
                "model_dir": args.protein_model_dir,
                "ckpt": args.ckpt,
                "inference_steps": args.inference_steps,
                "batch_size": args.batch_size,
            },
        )

    core_start = time.time()
    group_size = max(1, int(getattr(args, "pack_group_size", 12)))
    for start in range(0, len(chunk), group_size):
        group = chunk[start : start + group_size]
        owners: List[Entry] = []
        graphs: List[HeteroData] = []

        if getattr(args, "graph_build_workers", 0) > 0 and len(group) > 1:
            with ThreadPoolExecutor(max_workers=args.graph_build_workers) as executor:
                futures = [executor.submit(_build_protein_graph_single, entry, args) for entry in group]
                for future in as_completed(futures):
                    cid, built_graphs = future.result()
                    if built_graphs:
                        entry = next(item for item in group if item["complex_id"] == cid)
                        owners.append(entry)
                        graphs.extend(built_graphs)
        else:
            for entry in group:
                cid, built_graphs = _build_protein_graph_single(entry, args)
                if built_graphs:
                    owners.append(entry)
                    graphs.extend(built_graphs)

        if not graphs:
            continue

        written = _pack_batch_and_write(
            owners=owners,
            orig_graphs=graphs,
            copies_per_graph=int(getattr(args, "stage1_packing_num", 1)),
            model=protein_model,
            device=protein_device,
            side_schedule=side_schedule,
            t_to_sigma=protein_t2s,
            model_args=model_args,
            args=args,
        )

        by_cid: Dict[str, List[str]] = {}
        for path in written:
            cid = Path(path).parent.name
            by_cid.setdefault(cid, []).append(path)

        for entry in owners:
            entry["stage1_packed_proteins"] = by_cid.get(entry["complex_id"], [])
            old_manifest = load_manifest(entry["out_dir"], entry["complex_id"], "stage1", "core")
            current_paths = old_manifest.get("packed_proteins", [])
            current_paths.extend(entry["stage1_packed_proteins"])
            save_manifest(
                entry["out_dir"],
                entry["complex_id"],
                "stage1",
                "core",
                {"packed_proteins": current_paths},
            )

        _del_and_gc(graphs, owners, written)

    print(f"[GPU {gpu_local_id}] Stage 1 core elapsed: {time.time() - core_start:.2f}s")
    _del_and_gc(protein_model, protein_device, protein_t2s, model_args, side_schedule)
    return chunk


def _restore_and_relax_worker(
    entry: Entry,
    pocket_file: str,
    args_dict: Dict[str, Any],
    do_relax: bool = True,
    cuda_id: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    if cuda_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

    cid = entry["complex_id"]
    out_dir = entry["out_dir"]
    full_protein = entry["protein_path"]

    openmm_params = dict(
        ph=args_dict.get("openmm_ph", 7.0),
        explicit=args_dict.get("openmm_explicit", False),
        md_steps=args_dict.get("openmm_md_steps", 0),
        platform=args_dict.get("openmm_platform", "CPU"),
        k_restraint=args_dict.get("openmm_k_restraint", 0.0),
    )

    try:
        pocket_state = extract_protein_packing_state(os.path.abspath(pocket_file))
        restored = os.path.join(out_dir, f"{cid}_full_from_{Path(pocket_state).stem}.pdb")
        restore_full_protein_from_pocket(full_protein, pocket_state, restored, replace_backbone=False)

        relaxed = os.path.join(out_dir, f"{cid}_relaxed_from_{Path(pocket_state).stem}.pdb")
        if do_relax:
            out_path = openmm_sidechain_relax(
                restored,
                relaxed,
                ph=openmm_params["ph"],
                explicit=openmm_params["explicit"],
                md_steps=openmm_params["md_steps"],
                platform=openmm_params["platform"],
                k_restraint=openmm_params["k_restraint"],
            )
            return cid, out_path, None

        shutil.copyfile(restored, relaxed)
        return cid, relaxed, None
    except Exception as exc:
        try:
            if "restored" in locals() and os.path.exists(restored):
                return cid, restored, f"relax skipped or failed: {exc}"
        except Exception:
            pass
        return cid, full_protein, f"restore or relax failed: {exc}"


def _batch_restore_and_relax(
    entries: List[Entry],
    pocket_key: str,
    out_key: str,
    args: Namespace,
    max_workers: int = 8,
    max_inflight: Optional[int] = None,
    cuda_ids: Optional[List[int]] = None,
    do_relax: bool = True,
) -> List[Entry]:
    from threading import Semaphore

    if not entries:
        return entries

    nworkers = min(max_workers, max(1, len(entries)))
    if max_inflight is None:
        max_inflight = max(1, nworkers * 2)

    args_dict = dict(
        openmm_ph=getattr(args, "openmm_ph", 7.0),
        openmm_explicit=getattr(args, "openmm_explicit", False),
        openmm_md_steps=getattr(args, "openmm_md_steps", 0),
        openmm_platform=getattr(args, "openmm_platform", "CPU"),
        openmm_k_restraint=getattr(args, "openmm_k_restraint", 0.0),
    )

    semaphore = Semaphore(max_inflight)
    per_entry_results: Dict[str, List[str]] = {entry["complex_id"]: [] for entry in entries}
    jobs: List[Tuple[Entry, str, Dict[str, Any], bool, Optional[str]]] = []

    for entry in entries:
        pocket_files = entry.get(pocket_key, []) or []
        for pocket_file in pocket_files:
            jobs.append((entry, pocket_file, args_dict, do_relax, None))

    if not jobs:
        for entry in entries:
            entry[out_key] = []
        entries.sort(key=lambda d: d["complex_id"])
        return entries

    if cuda_ids:
        for i in range(len(jobs)):
            entry, pocket_file, job_args_dict, do_relax_flag, _ = jobs[i]
            jobs[i] = (entry, pocket_file, job_args_dict, do_relax_flag, str(cuda_ids[i % len(cuda_ids)]))

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = []
        for job in jobs:
            semaphore.acquire()
            future = executor.submit(_restore_and_relax_worker, *job)
            future.add_done_callback(lambda _future: semaphore.release())
            futures.append(future)

        for future in as_completed(futures):
            try:
                cid, output_path, err = future.result()
                if err:
                    print(f"[{cid}] WARNING: {_short(err)}")
                per_entry_results[cid].append(output_path)
            except Exception as exc:
                print(f"[batch_restore_relax] ERROR: {exc}")

    for entry in entries:
        entry[out_key] = per_entry_results.get(entry["complex_id"], [])

    entries.sort(key=lambda d: d["complex_id"])
    return entries


def stage1_postprocess(entries: List[Entry], args: Namespace) -> List[Entry]:
    entries = _batch_restore_and_relax(
        entries,
        pocket_key="stage1_packed_proteins",
        out_key="stage1_full_relaxed_proteins",
        args=args,
        max_workers=max(1, min(8, len(entries))),
        max_inflight=None,
        do_relax=bool(getattr(args, "relax_stage1", True)),
    )
    for entry in entries:
        save_manifest(
            entry["out_dir"],
            entry["complex_id"],
            "stage1",
            "postprocess",
            {
                "full_relaxed_proteins": entry.get("stage1_full_relaxed_proteins", []),
                "relax_enabled": bool(getattr(args, "relax_stage1", True)),
            },
        )
    return entries


# ---------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------
def stage2_preprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)

    for entry in chunk:
        out_dir = entry["out_dir"]
        cid = entry["complex_id"]
        prepared = {}
        packed = entry.get("stage1_full_relaxed_proteins", []) or entry.get("stage1_packed_proteins", [])

        for idx, packing_path in enumerate(packed):
            ligand_copy = copy_file(entry["ligand_path"], out_dir)
            try:
                protein_pdbqt, ligand_pdbqt = pdb2pdbqt(
                    os.path.abspath(packing_path),
                    ligand_copy,
                    remove_protein=False,
                    remove_ligand=True,
                )
                prepared[int(idx)] = {
                    "packing_path": packing_path,
                    "protein_pdbqt": protein_pdbqt,
                    "ligand_pdbqt": ligand_pdbqt,
                }
            except Exception as exc:
                print(f"[{cid}] WARNING (stage 2 preprocess): {_short(exc)}")

        entry["_stage2_prepared"] = prepared
        save_manifest(out_dir, cid, "stage2", "preprocess", {str(k): v for k, v in prepared.items()})

    return chunk


def stage2_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    try:
        if PACKDOCK_HOME and os.path.isdir(PACKDOCK_HOME):
            os.chdir(PACKDOCK_HOME)
    except Exception:
        pass

    max_workers = max(1, int(getattr(args, "docking_max_workers", 1)))
    max_inflight = getattr(args, "docking_max_inflight", None) or max_workers
    from threading import Semaphore

    semaphore = Semaphore(max_inflight)
    seed = getattr(args, "seed", "666")
    per_entry_states: Dict[str, Dict[int, List[str]]] = {entry["complex_id"]: {} for entry in chunk}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for entry in chunk:
            cid = entry["complex_id"]
            out_dir = entry["out_dir"]
            prepared = entry.get("_stage2_prepared", {})
            if not prepared:
                loaded = load_manifest(out_dir, cid, "stage2", "preprocess")
                prepared = {int(k): v for k, v in loaded.items()} if loaded else {}
                if prepared:
                    print(f"[{cid}] loaded stage 2 preprocess manifest ({len(prepared)} items)")

            for idx, meta in sorted(prepared.items(), key=lambda kv: int(kv[0])):
                semaphore.acquire()
                future = executor.submit(
                    stage2_core_one,
                    entry,
                    int(idx),
                    meta["protein_pdbqt"],
                    meta["ligand_pdbqt"],
                    seed,
                )
                future.add_done_callback(lambda _future: semaphore.release())
                futures.append(future)

        for future in as_completed(futures):
            try:
                cid, idx, states, err = future.result()
                if err:
                    print(f"[{cid}] WARNING (stage 2 core): {_short(err)}")
                if states:
                    per_entry_states[cid][idx] = states
            except Exception as exc:
                print(f"[stage 2 core pool] ERROR: {exc}")

    for entry in chunk:
        cid = entry["complex_id"]
        out_dir = entry["out_dir"]
        save_manifest(
            out_dir,
            cid,
            "stage2",
            "core",
            {"poses_by_idx": {str(k): v for k, v in per_entry_states.get(cid, {}).items()}},
        )
        entry["_stage2_core_states"] = per_entry_states.get(entry["complex_id"], {})

    return chunk


def stage2_postprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)

    for entry in chunk:
        cid = entry["complex_id"]
        out_dir = entry["out_dir"]
        original_ligand = entry["ligand_path"]
        per_job_states = entry.get("_stage2_core_states") or load_manifest(out_dir, cid, "stage2", "core").get("poses_by_idx", {})
        per_job_states = {int(k): v for k, v in per_job_states.items()} if per_job_states else {}

        docking_dfs: List[pd.DataFrame] = []
        for idx, states in sorted(per_job_states.items()):
            log_path = f"{out_dir}/{cid}_0_{idx}_docking_ligand_vina_score_out.log"
            df = read_first_docking_log(log_path)
            if df is None or df.empty:
                continue

            if getattr(args, "rmsd", False):
                rmsds: List[float] = []
                nums: List[int] = []
                for pose_k, ligand_path in enumerate(states):
                    rmsd_txt = f"{out_dir}/rmsd_loop0_idx{idx}_pose{pose_k}.txt"
                    os.system(f"obrms {original_ligand} {ligand_path} > {rmsd_txt}")
                    rmsd = read_rmsd_from_txt(rmsd_txt)
                    if rmsd is not None:
                        rmsds.append(rmsd)
                        nums.append(idx)
                if rmsds:
                    df = df.head(len(rmsds)).copy()
                    df["true_rmsd"] = rmsds
                    df["num"] = nums
            else:
                df = df.copy()
                df["num"] = idx

            docking_dfs.append(df)

        if docking_dfs:
            top_states, sorted_df = collect_top_docking(docking_dfs, cid, out_dir, top_n=5)
            score_path = f"{out_dir}/0_dockingscores.csv"
            sorted_df.to_csv(score_path, index=False)
            entry["stage2_top_ligand_states"] = top_states
            save_manifest(
                out_dir,
                cid,
                "stage2",
                "postprocess",
                {"dockingscores_csv": score_path, "top_states": top_states},
            )
        else:
            entry["stage2_top_ligand_states"] = []
            save_manifest(
                out_dir,
                cid,
                "stage2",
                "postprocess",
                {"dockingscores_csv": None, "top_states": []},
            )

        if "_stage2_core_states" in entry:
            del entry["_stage2_core_states"]
        if "_stage2_prepared" in entry:
            del entry["_stage2_prepared"]

    return chunk


# ---------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------
def stage3_preprocess_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    complex_model, complex_device, complex_t2s, model_args = load_complex_model(args)
    side_schedule = get_t_schedule(inference_steps=args.inference_steps)

    for entry in chunk:
        save_manifest(
            entry["out_dir"],
            entry["complex_id"],
            "stage3",
            "preprocess",
            {"model_dir": args.model_dir, "ckpt": args.ckpt},
        )

    core_start = time.time()
    group_size = max(1, int(getattr(args, "pack_group_size", 12)))
    for start in range(0, len(chunk), group_size):
        group = chunk[start : start + group_size]
        owners: List[Entry] = []
        graphs: List[HeteroData] = []

        if getattr(args, "graph_build_workers", 0) > 0 and len(group) > 1:
            with ThreadPoolExecutor(max_workers=args.graph_build_workers) as executor:
                futures = [executor.submit(_build_complex_graphs_single, entry, args) for entry in group]
                for future in as_completed(futures):
                    cid, built_graphs = future.result()
                    if built_graphs:
                        entry = next(item for item in group if item["complex_id"] == cid)
                        owners.extend([entry] * len(built_graphs))
                        graphs.extend(built_graphs)
        else:
            for entry in group:
                cid, built_graphs = _build_complex_graphs_single(entry, args)
                if built_graphs:
                    owners.extend([entry] * len(built_graphs))
                    graphs.extend(built_graphs)

        if not graphs:
            for entry in group:
                if not entry.get("stage2_top_ligand_states", []):
                    entry["stage3_packed_complex"] = []
            continue

        written = _pack_batch_and_write(
            owners=owners,
            orig_graphs=graphs,
            copies_per_graph=int(getattr(args, "stage3_packing_num", 5)),
            model=complex_model,
            device=complex_device,
            side_schedule=side_schedule,
            t_to_sigma=complex_t2s,
            model_args=model_args,
            args=args,
        )

        by_cid: Dict[str, List[str]] = {}
        for path in written:
            cid = Path(path).parent.name
            by_cid.setdefault(cid, []).append(path)

        for entry in group:
            entry["stage3_packed_complex"] = by_cid.get(entry["complex_id"], [])
            old_manifest = load_manifest(entry["out_dir"], entry["complex_id"], "stage3", "core")
            current_paths = old_manifest.get("packed_complex", [])
            current_paths.extend(entry["stage3_packed_complex"])
            save_manifest(
                entry["out_dir"],
                entry["complex_id"],
                "stage3",
                "core",
                {"packed_complex": current_paths},
            )

        _del_and_gc(graphs, owners, written)

    print(f"[GPU {gpu_local_id}] Stage 3 core elapsed: {time.time() - core_start:.2f}s")
    _del_and_gc(complex_model, complex_device, complex_t2s, model_args, side_schedule)
    return chunk


def stage3_postprocess(entries: List[Entry], args: Namespace) -> List[Entry]:
    entries = _batch_restore_and_relax(
        entries,
        pocket_key="stage3_packed_complex",
        out_key="stage3_full_relaxed_proteins",
        args=args,
        max_workers=max(1, min(8, len(entries))),
        max_inflight=None,
        do_relax=bool(getattr(args, "relax_stage3", True)),
    )
    for entry in entries:
        save_manifest(
            entry["out_dir"],
            entry["complex_id"],
            "stage3",
            "postprocess",
            {
                "full_relaxed_proteins": entry.get("stage3_full_relaxed_proteins", []),
                "relax_enabled": bool(getattr(args, "relax_stage3", True)),
            },
        )
    return entries


# ---------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------
def stage4_preprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)

    for entry in chunk:
        out_dir = entry["out_dir"]
        cid = entry["complex_id"]
        prepared = {}
        packed = entry.get("stage3_full_relaxed_proteins", []) or entry.get("stage3_packed_complex", [])

        for idx, packing_path in enumerate(packed):
            ligand_copy = copy_file(entry["ligand_path"], out_dir)
            try:
                protein_pdbqt, ligand_pdbqt = pdb2pdbqt(
                    os.path.abspath(packing_path),
                    ligand_copy,
                    remove_protein=False,
                    remove_ligand=True,
                )
                prepared[int(idx)] = {
                    "packing_path": packing_path,
                    "protein_pdbqt": protein_pdbqt,
                    "ligand_pdbqt": ligand_pdbqt,
                }
            except Exception as exc:
                print(f"[{cid}] WARNING (stage 4 preprocess): {_short(exc)}")

        entry["_stage4_prepared"] = prepared
        save_manifest(out_dir, cid, "stage4", "preprocess", {str(k): v for k, v in prepared.items()})

    return chunk


def stage4_core_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)
    try:
        if PACKDOCK_HOME and os.path.isdir(PACKDOCK_HOME):
            os.chdir(PACKDOCK_HOME)
    except Exception:
        pass

    max_workers = max(1, int(getattr(args, "docking_max_workers", 1)))
    max_inflight = getattr(args, "docking_max_inflight", None) or max_workers
    from threading import Semaphore

    semaphore = Semaphore(max_inflight)
    seed = getattr(args, "seed", "666")
    per_entry_states: Dict[str, Dict[int, List[str]]] = {entry["complex_id"]: {} for entry in chunk}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for entry in chunk:
            cid = entry["complex_id"]
            out_dir = entry["out_dir"]

            prepared = entry.get("_stage4_prepared", {})
            if not prepared:
                loaded = load_manifest(out_dir, cid, "stage4", "preprocess")
                prepared = {int(k): v for k, v in loaded.items()} if loaded else {}
                if prepared:
                    print(f"[{cid}] loaded stage 4 preprocess manifest ({len(prepared)} items)")

            for idx, meta in sorted(prepared.items(), key=lambda kv: int(kv[0])):
                semaphore.acquire()
                future = executor.submit(
                    stage4_core_one,
                    entry,
                    int(idx),
                    meta["protein_pdbqt"],
                    meta["ligand_pdbqt"],
                    seed,
                )
                future.add_done_callback(lambda _future: semaphore.release())
                futures.append(future)

        for future in as_completed(futures):
            try:
                cid, idx, states, err = future.result()
                if err:
                    print(f"[{cid}] WARNING (stage 4 core): {_short(err)}")
                if states:
                    per_entry_states[cid][idx] = states
            except Exception as exc:
                print(f"[stage 4 core pool] ERROR: {exc}")

    for entry in chunk:
        cid = entry["complex_id"]
        out_dir = entry["out_dir"]
        poses = per_entry_states.get(cid, {})
        save_manifest(
            out_dir,
            cid,
            "stage4",
            "core",
            {"poses_by_idx": {str(k): v for k, v in poses.items()}},
        )
        entry["_stage4_core_states"] = poses

    return chunk


def stage4_postprocess_worker(chunk: List[Entry], args: Namespace, gpu_local_id: int) -> List[Entry]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_id)

    for entry in chunk:
        cid = entry["complex_id"]
        out_dir = entry["out_dir"]
        original_ligand = entry["ligand_path"]

        per_job_states = entry.get("_stage4_core_states")
        if not per_job_states:
            loaded = load_manifest(out_dir, cid, "stage4", "core")
            per_job_states = loaded.get("poses_by_idx", {}) if loaded else {}
        per_job_states = {int(k): v for k, v in per_job_states.items()} if per_job_states else {}

        docking_dfs: List[pd.DataFrame] = []
        for idx, states in sorted(per_job_states.items()):
            log_path = f"{out_dir}/{cid}_1_{idx}_docking_ligand_vina_score_out.log"
            df = read_first_docking_log(log_path)
            if df is None or df.empty:
                continue

            if getattr(args, "rmsd", False):
                rmsds: List[float] = []
                nums: List[int] = []
                for pose_k, ligand_path in enumerate(states):
                    try:
                        rmsd_txt = f"{out_dir}/rmsd_loop1_idx{idx}_pose{pose_k}.txt"
                        os.system(f"obrms {original_ligand} {ligand_path} > {rmsd_txt}")
                        rmsd = read_rmsd_from_txt(rmsd_txt)
                        if rmsd is not None:
                            rmsds.append(rmsd)
                            nums.append(idx)
                    except Exception as exc:
                        print(f"[{cid}] WARNING: RMSD calculation failed for idx={idx}, pose={pose_k}: {_short(exc)}")

                if rmsds:
                    df = df.head(len(rmsds)).copy()
                    df["true_rmsd"] = rmsds
                    df["num"] = nums
                else:
                    df = df.copy()
                    df["num"] = idx
            else:
                df = df.copy()
                df["num"] = idx

            docking_dfs.append(df)

        if docking_dfs:
            merged = pd.concat(docking_dfs, ignore_index=True)
            score_path = f"{out_dir}/1_dockingscores.csv"
            if "true_rmsd" in merged.columns:
                merged.sort_values("true_rmsd").to_csv(score_path, index=False)
            else:
                merged.sort_values("affinity").to_csv(score_path, index=False)

            save_manifest(
                out_dir,
                cid,
                "stage4",
                "postprocess",
                {
                    "dockingscores_csv": score_path,
                    "poses_by_idx": {str(k): v for k, v in per_job_states.items()},
                },
            )
        else:
            save_manifest(
                out_dir,
                cid,
                "stage4",
                "postprocess",
                {
                    "dockingscores_csv": None,
                    "poses_by_idx": {str(k): v for k, v in per_job_states.items()},
                },
            )

        if "_stage4_core_states" in entry:
            del entry["_stage4_core_states"]

    return chunk


# ---------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------
def stage0_prepare_and_pocket(args: Namespace) -> List[Entry]:
    entries = prepare_entries(args)
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(entries)))) as executor:
        futures = [executor.submit(try_extract_pocket_robust, entry, args) for entry in entries]
        entries = [future.result() for future in as_completed(futures)]
    entries.sort(key=lambda d: d["complex_id"])
    return entries


def main() -> None:
    args = parse_arguments()

    print("Stage 0: preparing entries and extracting pockets...")
    entries = stage0_prepare_and_pocket(args)
    print(f"Prepared {len(entries)} entries")

    print("Stage 1 preprocess: loading protein model and schedule...")
    print("Stage 1 core: protein packing on empty pockets...")
    entries = run_per_gpu(entries, stage1_preprocess_core_worker, args)
    print("Stage 1 postprocess: restoring full receptors and optional relaxation...")
    entries = stage1_postprocess(entries, args)

    print("Stage 2 preprocess: copying ligands and generating PDBQT files...")
    entries = run_per_gpu(entries, stage2_preprocess_worker, args)
    print("Stage 2 core: docking against stage 1 conformations...")
    entries = run_per_gpu(entries, stage2_core_worker, args)
    print("Stage 2 postprocess: parsing docking logs and selecting top poses...")
    entries = run_per_gpu(entries, stage2_postprocess_worker, args)

    print("Stage 3 preprocess: loading complex model and schedule...")
    print("Stage 3 core: complex packing using top poses...")
    entries = run_per_gpu(entries, stage3_preprocess_core_worker, args)
    print("Stage 3 postprocess: restoring full receptors and optional relaxation...")
    entries = stage3_postprocess(entries, args)

    print("Stage 4 preprocess: copying ligands and generating PDBQT files...")
    entries = run_per_gpu(entries, stage4_preprocess_worker, args)
    print("Stage 4 core: final docking on packed complexes...")
    entries = run_per_gpu(entries, stage4_core_worker, args)
    print("Stage 4 postprocess: parsing docking logs and writing final scores...")
    entries = run_per_gpu(entries, stage4_postprocess_worker, args)

    print(f"All entries processed successfully.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
