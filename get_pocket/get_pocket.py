import pickle
import sys
import os
import glob
import time
import multiprocessing
from multiprocessing import Pool

import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from scipy import sparse
from scipy.spatial import distance_matrix
from Bio.PDB import *
from Bio.PDB.PDBIO import Select

import shutil

import warnings
warnings.filterwarnings('ignore')

def extract(ligand, pdb, key):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb)
    ligand_positions = ligand.GetConformer().GetPositions()
    # Get distance between ligand positions (N_ligand, 3) and
    # residue positions (N_residue, 3) for each residue
    # only select residue with minimum distance of it is smaller than 5A
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            residue_positions = np.array([np.array(list(atom.get_vector())) \
                for atom in residue.get_atoms()])    # if "H" not in atom.get_id()
            if len(residue_positions.shape) < 2:
                print(residue)
                return 0
            min_dis = np.min(distance_matrix(residue_positions, ligand_positions))
            if min_dis < 12.0:
                return 1
            else:
                return 0
    
    io = PDBIO()
    io.set_structure(structure)
    fn = "BS_tmp_"+str(key)+".pdb"
    io.save(fn, ResidueSelect())
    try:
        m2 = Chem.MolFromPDBFile(fn)#,removeHs=False)
        #may contain metal atom, causing MolFromPDBFile return None
        if m2 is None:
            print("first read PDB fail",fn)
            remove_zn_dir=f"{pocket_dir}/docker_result_remove_ZN"
            if not os.path.exists(remove_zn_dir):
                os.mkdir(remove_zn_dir)
            cmd=f"cp {fn}   {remove_zn_dir}"
            print(cmd)
            os.system(cmd)

            fn_remove_zn=os.path.join(remove_zn_dir,fn.replace('.pdb','_remove_ZN.pdb'))
            cmd=f"sed -e '/ZN/d'  {fn}  > {fn_remove_zn}"
            os.system(cmd)
            print("delete metal atom and get new pdb file",fn_remove_zn)
            m2 = Chem.MolFromPDBFile(fn_remove_zn)#,removeHs=False)
        else:
            os.system("rm -f " + fn)
    except:
        print("Read PDB fail for other unknow reason",fn)


    return m2

def preprocessor(ligand_pdb,receptor_fn):
    '''
    Extract the pocket area from the docking result and persist it to the file :(m1,m2)
    Input: interface result sdf file and original protein PDB file;
    Output: Extract the pocket area and save it in the {data_dir}/sdf_fn file
    '''
    ligand_id = ligand_pdb.split("/")[-1].split("_")[0] 
    pdbid = receptor_fn.split("/")[-2]
    key = f"{pdbid}_{ligand_id}"
    data_dir = f"{pocket_dir}/{pdbid}_{ligand_id}"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(ligand_pdb)
    ligand = Chem.MolFromPDBFile(ligand_pdb, sanitize=False, removeHs=False)

    try:
        m2 = extract(ligand, receptor_fn,key)
        PDBwriter = Chem.PDBWriter(f'{data_dir}/protein_pocket.pdb')
        PDBwriter.write(m2)
        PDBwriter.close()
        src_ligand_pdb = ligand_pdb
        dst_ligand = f"{data_dir}/ligand.pdb" 
        shutil.copyfile(src_ligand_pdb, dst_ligand)
        with open(os.path.join(data_dir,key), "wb") as fp:
            pickle.dump((ligand, m2), fp, pickle.HIGHEST_PROTOCOL)   #binary file
    except:
        print(f'extract m2 failed {pdbid}_{ligand_id}')
        # os.remove(data_dir)
        # continue

    return 0


if __name__ == '__main__':

    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
    import time
    from multiprocessing import Pool, cpu_count
    import os
    import pandas as pd
    group = "group3"
    apolist = pd.read_csv(f"./apo2holo_datasets/{group}.list")
    total_num = len(apolist["PDBID"])
    # i = 0
    file_tuple_list = []
    apo_tuple_list = []
    pocket_dir = f"./Apo_Holo_fulldata_aligned_pocket_{group}_12a_new"
    for i, apo_id in enumerate(apolist['PDBID']):
        apo_dir = f"./apo2holo_datasets/{group}/{apo_id}"
        ligand_dir = f"./apo2holo_datasets/{group}/{apo_id}/Ligands"
        apo_receptor_fn=os.path.join(apo_dir + '/apo_aligned.pdb')
        holo_receptor_fn=os.path.join(apo_dir + '/template_aligned.pdb')
        ligand_pdb_fn_list = glob.glob(os.path.join(ligand_dir, '*.pdb'))
        for ligand_pdb_fn in ligand_pdb_fn_list:
            apo_tuple_list.append((ligand_pdb_fn, apo_receptor_fn))
    for file_tuple in apo_tuple_list:
        preprocessor(file_tuple[0],file_tuple[1])
    

    print("all pocket done! check the outdir plz/sdf/pdb!")
