import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles


def t_to_sigma(t_side, args):
    side_sigma = args.side_sigma_min ** (1-t_side) * args.side_sigma_max ** t_side
    return side_sigma


def modify_conformer(data, side_updates):
    recptor_center = torch.mean(data['atom'].pos, dim=0, keepdim=True)
    rigid_pos = data['atom'].pos
    flexible_pos = modify_conformer_torsion_angles(rigid_pos,
                                                        data['atom', 'atom_contact', 'atom'].edge_index.T[data.tor_mask],
                                                        data.mask_rotate if isinstance(data.mask_rotate, np.ndarray) else data.mask_rotate[0],
                                                        side_updates).to(rigid_pos.device)
    data['atom'].pos = flexible_pos

    return data


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def set_time(complex_graphs, t, batchsize, all_atoms, device):
    def _assign(ntype: str):
        # Skip atoms when not requested
        # if ntype == 'atom' and not all_atoms:
        #     return
        # Only proceed if this node type is actually in the graph
        if ntype in complex_graphs.node_types:
            n = complex_graphs[ntype].num_nodes
            if n > 0:
                complex_graphs[ntype].node_t = {
                    't': torch.full((n,), fill_value=t, device=device)
                }

    _assign('receptor')
    _assign('ligand')
    _assign('atom')

    complex_graphs.complex_t = {
        't': torch.full((batchsize,), fill_value=t, device=device)
    }
