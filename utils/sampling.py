import numpy as np
import torch
from torch_geometric.loader import DataLoader,DataListLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R


def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
    for complex_graph in data_list:
        side_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph.tor_mask.sum())
        complex_graph['atom'].pos = \
            modify_conformer_torsion_angles(complex_graph['atom'].pos,
                                            complex_graph['atom', 'atom_contact', 'atom'].edge_index.T[complex_graph.tor_mask],
                                            complex_graph.mask_rotate if isinstance(complex_graph.mask_rotate, np.ndarray) else complex_graph.mask_rotate[0],
                                            side_updates)


def sampling(data_list, model, inference_steps, side_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):


        t_side = side_schedule[t_idx]
        dt_side = side_schedule[t_idx] - side_schedule[t_idx + 1] if t_idx < inference_steps - 1 else side_schedule[t_idx]


        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            side_sigma = t_to_sigma(t_side)

            set_time(complex_graph_batch, t_side, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                side_score = model(complex_graph_batch)
            side_g = side_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.side_sigma_max / model_args.side_sigma_min)))

            if ode:
                side_perturb = (0.5 * side_g ** 2 * dt_side * side_score.cpu()).numpy()
            else:
                side_z = torch.zeros(side_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=side_score.shape)
                side_perturb = (side_g ** 2 * dt_side * side_score.cpu() + side_g * np.sqrt(dt_side) * side_z).numpy()
            torsions_per_protein = side_perturb.shape[0] // b
            new_data_list.extend([modify_conformer(complex_graph,side_perturb[i * torsions_per_protein:(i + 1) * torsions_per_protein])
                        for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['atom'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence
