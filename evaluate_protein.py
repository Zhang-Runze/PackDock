import copy
import os
import torch
import time
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import wandb
from biopandas.pdb import PandasPdb
from rdkit import RDLogger
from torch_geometric.loader import DataLoader,DataListLoader

from datasets.pdbbind import PDBBind, PDBprotein, read_mol, read_pdb, read_protein_pdb
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, get_symmetry_rmsd, remove_all_hs, read_strings_from_txt, kabsch_rmsd, ExponentialMovingAverage
from utils.visualise import PDBFile
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
import yaml

cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--model_dir', type=str, default='workdir/protein_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
parser.add_argument('--run_name', type=str, default='test', help='')
parser.add_argument('--project', type=str, default='ligbind_inf', help='')
parser.add_argument('--out_dir', type=str, default=None, help='Where to save results to')
parser.add_argument('--batch_size', type=int, default=10, help='Number of poses to sample in parallel')
parser.add_argument('--cache_path', type=str, default='data/cacheNew', help='Folder from where to load/restore cached dataset')
parser.add_argument('--data_dir', type=str, default='data/PDBBind_processed/', help='Folder containing original structures')
parser.add_argument('--split_path', type=str, default='data/splits/timesplit_no_lig_overlap_val', help='Path of file defining the split')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--limit_complexes', type=int, default=0, help='Limit to the number of complexes')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Whether to save visualizations')
parser.add_argument('--samples_per_complex', type=int, default=1, help='Number of poses to sample for each complex')
parser.add_argument('--actual_steps', type=int, default=None, help='')
args = parser.parse_args()



if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value

if args.out_dir is None: args.out_dir = f'inference_out_dir_protein_not_specified/{args.run_name}'
os.makedirs(args.out_dir, exist_ok=True)
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))


# names_no_rec_overlap = read_strings_from_txt(f'data/splits/bc40_test_set')
names_no_rec_overlap = read_strings_from_txt(f'data/splits/CASP14_list')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_dataset = PDBprotein(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                       receptor_radius=score_model_args.receptor_radius,
                       cache_path=args.cache_path, split_path=args.split_path,
                       remove_hs=score_model_args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                       matching=not score_model_args.no_torsion, keep_original=True,
                       popsize=score_model_args.matching_popsize,
                       maxiter=score_model_args.matching_maxiter,
                       all_atoms=score_model_args.all_atoms,
                       atom_radius=score_model_args.atom_radius,
                       atom_max_neighbors=score_model_args.atom_max_neighbors,
                       esm_embeddings_path=score_model_args.esm_embeddings_path,
                        # esm_embeddings_path="datasets/test_no_rec_overlap_pocket_emb_dictlist.pkl",
                       require_ligand=True,
                       num_workers=args.num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)



t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

if not args.no_model:
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, only_protein=True)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    if args.ckpt == 'last_model.pt':
        model_state_dict = state_dict['model']
        ema_weights_state = state_dict['ema_weights']
        model.load_state_dict(model_state_dict, strict=True)
        ema_weights = ExponentialMovingAverage(model.parameters(), decay=score_model_args.ema_rate)
        ema_weights.load_state_dict(ema_weights_state, device=device)
        ema_weights.copy_to(model.parameters())
    else:
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

    confidence_model = None
    confidence_args = None
    confidence_model_args = None


side_schedule = get_t_schedule(inference_steps=args.inference_steps)
print('t schedule', side_schedule)

rmsds_list, obrmsds, centroid_distances_list, failures, min_cross_distances_list, base_min_cross_distances_list, confidences_list, names_list = [], [], [], 0, [], [], [], []
run_times, min_self_distances_list, without_rec_overlap_list = [], [], []
N = args.samples_per_complex

print('Size of test dataset: ', len(test_dataset))
rmsd_dict = {}
for idx, orig_complex_graph in tqdm(enumerate(test_loader)):

    success = 0
    while not success:

        success = 1
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(data_list, score_model_args.no_torsion, args.no_random, score_model_args.tr_sigma_max)

        pdb = None
        if args.save_visualisation:
            visualization_list = []
            for idx, graph in enumerate(data_list):
                mol = read_protein_pdb(args.data_dir, graph['name'][0], remove_hs=score_model_args.remove_hs)
                pdb = PDBFile(mol)
                pdb.add(mol, 0, 0)
                pdb.add((orig_complex_graph['atom'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                pdb.add((graph['atom'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)
        else:
            visualization_list = None

        start_time = time.time()
        if not args.no_model:

            confidence_data_list = None

            data_list, confidence = sampling(data_list=data_list, model=model,
                                                inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                side_schedule=side_schedule, 
                                                device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                no_random=args.no_random,
                                                ode=args.ode, visualization_list=visualization_list,
                                                confidence_model=confidence_model,
                                                confidence_data_list=confidence_data_list,
                                                confidence_model_args=confidence_model_args,
                                                batch_size=args.batch_size,
                                                no_final_step_noise=args.no_final_step_noise)

        run_times.append(time.time() - start_time)
        if score_model_args.no_torsion: orig_complex_graph['atom'].orig_pos = orig_complex_graph['atom'].pos.cpu().numpy()

        filterHs = torch.not_equal(data_list[0]['atom'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['atom'].orig_pos, list):
            orig_complex_graph['atom'].orig_pos = orig_complex_graph['atom'].orig_pos[0]

        side_pos = np.asarray(
            [complex_graph['atom'].pos.cpu().numpy()[filterHs] for complex_graph in data_list])
        orig_side_pos = np.expand_dims(orig_complex_graph['atom'].orig_pos[filterHs],axis=0)

        rotated_atom_num = orig_complex_graph.rotated_atom_num
        rmsd = kabsch_rmsd(orig_side_pos, side_pos, N, rotated_atom_num)
        dict_key = orig_complex_graph['name'][0]
        rmsd_dict[dict_key]=rmsd
        rmsds_list.append(rmsd)

        print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1))
        self_distances = np.linalg.norm(side_pos[:, :, None, :] - side_pos[:, None, :, :], axis=-1)
        self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
        min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))


        if args.save_visualisation:
            for rank, batch_idx in enumerate(np.argsort(rmsd)):
                visualization_list[batch_idx].write(
                    f'{args.out_dir}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}.pdb')
        without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)
        names_list.append(orig_complex_graph.name[0])


print('Performance without hydrogens included in the loss')
print(failures, "failures due to exceptions")
rmsd_df = pd.DataFrame(rmsd_dict)
rmsd_df.to_csv("rmsd_CASP14.csv",index = False)


