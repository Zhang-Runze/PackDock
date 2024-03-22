import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

"""
    Preprocessing and computation for torsional updates to conformers
"""
def get_side (G2, backbone_idx):
    """Find the connected graph that's not on the skeleton"""
    G2_ = list(nx.connected_components(G2))
    G0 = G2_[0]
    G0_list = list(G0)
    i = G0_list[0]
    if backbone_idx[i]==1 or len(G0) > 10 :
        return list(G2_[1])
    else:
        return list(G2_[0])


def get_transformation_mask(pyg_data, tor_idx, backbone_idx,pocket_outside_idx):
    backbone_list = backbone_idx.tolist()
    pocket_outside_list = pocket_outside_idx.tolist()
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = tor_idx.T.numpy()


    for i in range(0, edges.shape[0]):
        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        u, v = edges[i]
        if not nx.is_connected(G2):
            all_connected_compoents = sorted(nx.connected_components(G2), key=len)
            num_connected_compoents = len(all_connected_compoents)
            for num_G in range(num_connected_compoents):
                l = list(all_connected_compoents[num_G])
                result_list = [backbone_list[i] == 1 for i in l] 
                result_pocket_list = [pocket_outside_list[i] == 1 for i in l] 
                if 1 in result_list:
                    continue
                elif 1 in result_pocket_list:
                    continue
                elif u not in l and v not in l:
                    continue
                else:
                    break
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    assert np.sum(mask_edges) != 0 
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(to_rotate)):
        if to_rotate[i] != []:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1
    assert mask_rotate.shape[0] == np.sum(mask_edges)
    return mask_edges, mask_rotate




def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]
        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer_torsion_angles(data.pos,
                                               data.edge_index.T[data.edge_mask],
                                               data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new