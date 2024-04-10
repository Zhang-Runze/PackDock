import math
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn.nn import BatchNorm




from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type= None):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:   # performance can be expanded through embedding of the protein language model
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())


        if self.num_scalar_features > 0:
            a = x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features]
            a = a.to(torch.float32)
            x_embedding += self.linear(a)

        
        if self.lm_embedding_type is not None:
            esm = x[:, -self.lm_embedding_dim:]
            esm = esm.to(torch.float32)
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, esm], axis=1))
        
        return x_embedding


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=False, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(TensorProductScoreModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.num_conv_layers = num_conv_layers
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        # embedding layers
        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.atom_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.atom_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lr_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.ar_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.la_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
        self.atom_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        # convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            for _ in range(9): # 3 intra & 6 inter per each layer
                conv_layers.append(TensorProductConvLayer(**parameters))

        self.conv_layers = nn.ModuleList(conv_layers)




        self.side_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
        self.tor_bond_conv = TensorProductConvLayer(
            in_irreps=self.conv_layers[-1].out_irreps,
            sh_irreps=self.final_tp_tor.irreps_out,
            out_irreps=f'{ns}x0o + {ns}x0e',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.tor_final_layer = nn.Sequential(
                nn.Linear(2 * ns, ns, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False)
            )


    def forward(self, data):
        if not self.confidence_mode:
            tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['t']])
        else:
            tor_sigma = [data.complex_t[noise_type] for noise_type in ['t']]

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build atom graph
        atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(data)
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

        # build cross graph
        cross_cutoff = (data.complex_t["t"] * 10 + 5).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance
        lr_edge_index, lr_edge_attr, lr_edge_sh, la_edge_index, la_edge_attr, \
            la_edge_sh, ar_edge_index, ar_edge_attr, ar_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        lr_edge_attr= self.lr_edge_embedding(lr_edge_attr)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)
        ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)

        for l in range(self.num_conv_layers):
            # ATOM UPDATES

            atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[atom_edge_index[0], :self.ns], atom_node_attr[atom_edge_index[1], :self.ns]], -1)
            atom_update = self.conv_layers[9*l](atom_node_attr, atom_edge_index, atom_edge_attr_, atom_edge_sh)

            al_edge_attr_ = torch.cat([la_edge_attr, atom_node_attr[la_edge_index[1], :self.ns], lig_node_attr[la_edge_index[0], :self.ns]], -1)
            al_update = self.conv_layers[9*l+1](lig_node_attr, torch.flip(la_edge_index, dims=[0]), al_edge_attr_,
                                                la_edge_sh, out_nodes=atom_node_attr.shape[0])

            ar_edge_attr_ = torch.cat([ar_edge_attr, atom_node_attr[ar_edge_index[0], :self.ns], rec_node_attr[ar_edge_index[1], :self.ns]],-1)
            ar_update = self.conv_layers[9*l+2](rec_node_attr, ar_edge_index, ar_edge_attr_, ar_edge_sh, out_nodes=atom_node_attr.shape[0])


            if l != self.num_conv_layers-1:  # last layer optimisation

                # LIGAND updates
                lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_edge_index[0], :self.ns], lig_node_attr[lig_edge_index[1], :self.ns]], -1)
                lig_update = self.conv_layers[9*l+3](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

                lr_edge_attr_ = torch.cat([lr_edge_attr, lig_node_attr[lr_edge_index[0], :self.ns], rec_node_attr[lr_edge_index[1], :self.ns]], -1)
                lr_update = self.conv_layers[9*l+4](rec_node_attr, lr_edge_index, lr_edge_attr_, lr_edge_sh,
                                                    out_nodes=lig_node_attr.shape[0])

                la_edge_attr_ = torch.cat([la_edge_attr, lig_node_attr[la_edge_index[0], :self.ns], atom_node_attr[la_edge_index[1], :self.ns]], -1)
                la_update = self.conv_layers[9*l+5](atom_node_attr, la_edge_index, la_edge_attr_, la_edge_sh,
                                                out_nodes=lig_node_attr.shape[0])


                # RECEPTOR updates
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_edge_index[0], :self.ns], rec_node_attr[rec_edge_index[1], :self.ns]], -1)
                rec_update = self.conv_layers[9*l+6](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

                rl_edge_attr_ = torch.cat([lr_edge_attr, rec_node_attr[lr_edge_index[1], :self.ns], lig_node_attr[lr_edge_index[0], :self.ns]], -1)
                rl_update = self.conv_layers[9*l+7](lig_node_attr, torch.flip(lr_edge_index, dims=[0]), rl_edge_attr_,
                                                    lr_edge_sh, out_nodes=rec_node_attr.shape[0])

                ra_edge_attr_ = torch.cat([ar_edge_attr, rec_node_attr[ar_edge_index[1], :self.ns], atom_node_attr[ar_edge_index[0], :self.ns]], -1)
                ra_update = self.conv_layers[9*l+8](atom_node_attr, torch.flip(ar_edge_index, dims=[0]), ra_edge_attr_,
                                                    ar_edge_sh, out_nodes=rec_node_attr.shape[0])

            # padding original features and update features with residual updates
            atom_node_attr = F.pad(atom_node_attr, (0, atom_update.shape[-1] - rec_node_attr.shape[-1]))
            atom_node_attr = atom_node_attr + atom_update + al_update + ar_update
            

            if l != self.num_conv_layers - 1:  # last layer optimisation
                lig_node_attr = F.pad(lig_node_attr, (0, lig_update.shape[-1] - lig_node_attr.shape[-1]))
                lig_node_attr = lig_node_attr + lig_update + la_update + lr_update
                rec_node_attr = F.pad(rec_node_attr, (0, rec_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_update + ra_update + rl_update

        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_side_bond_conv_graph(data)
        tor_bond_vec = data['atom'].pos[tor_bonds[1]] - data['atom'].pos[tor_bonds[0]]
        tor_bond_attr = atom_node_attr[tor_bonds[0]] + atom_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, atom_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)

        tor_pred = self.tor_bond_conv(atom_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                      out_nodes=data.tor_mask.sum(), reduce='mean')

        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['atom'].batch][data['atom', 'atom_contact', 'atom'].edge_index[0]][data.tor_mask]

        tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['atom'].x.device))

        return tor_pred

    def build_lig_conv_graph(self, data):
        # build the graph between ligand atoms
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['t'])

        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # build the graph between receptor residues
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['t'])
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)

        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_atom_conv_graph(self, data):
        # build the graph between receptor atoms
        data['atom'].node_sigma_emb = self.timestep_emb_func(data['atom'].node_t['t'])
        node_attr = torch.cat([data['atom'].x, data['atom'].node_sigma_emb], 1)
        radius_edges = radius_graph(data['atom'].pos, self.lig_max_radius, data['atom'].batch)
        edge_index = torch.cat([data['atom', 'atom_contact', 'atom'].edge_index, radius_edges], 1).long()
        edge_index = torch.unique(edge_index.T, dim=0)
        edge_index = edge_index.T
        src, dst = edge_index
        edge_vec = data['atom'].pos[dst.long()] - data['atom'].pos[src.long()]

        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['atom'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # LIGAND to RECEPTOR
        if torch.is_tensor(lr_cross_distance_cutoff):
            # different cutoff for every graph
            lr_edge_index = radius(data['receptor'].pos / lr_cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / lr_cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            lr_edge_index = radius(data['receptor'].pos, data['ligand'].pos, lr_cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        lr_edge_vec = data['receptor'].pos[lr_edge_index[1].long()] - data['ligand'].pos[lr_edge_index[0].long()]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data['ligand'].node_sigma_emb[lr_edge_index[0].long()]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

        cutoff_d = lr_cross_distance_cutoff[data['ligand'].batch[lr_edge_index[0]]].squeeze() \
            if torch.is_tensor(lr_cross_distance_cutoff) else lr_cross_distance_cutoff

        # LIGAND to ATOM
        la_edge_index = radius(data['atom'].pos, data['ligand'].pos, self.lig_max_radius,
                               data['atom'].batch, data['ligand'].batch, max_num_neighbors=10000)

        la_edge_vec = data['atom'].pos[la_edge_index[1].long()] - data['ligand'].pos[la_edge_index[0].long()]
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data['ligand'].node_sigma_emb[la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization='component')

        # ATOM to RECEPTOR
        ar_edge_index = data['atom', 'receptor'].edge_index
        ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['atom'].pos[ar_edge_index[0].long()]
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data['atom'].node_sigma_emb[ar_edge_index[0].long()]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')

        return lr_edge_index, lr_edge_attr, lr_edge_sh, la_edge_index, la_edge_attr, \
               la_edge_sh, ar_edge_index, ar_edge_attr, ar_edge_sh


    def build_side_bond_conv_graph(self, data):
        bonds = data['atom', 'atom_contact', 'atom'].edge_index[:, data.tor_mask].long()
        bond_pos = (data['atom'].pos[bonds[0]] + data['atom'].pos[bonds[1]]) / 2
        bond_batch = data['atom'].batch[bonds[0]]
        edge_index = radius(data['atom'].pos, bond_pos, self.lig_max_radius, batch_x=data['atom'].batch, batch_y=bond_batch)
        
        edge_vec = data['atom'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.atom_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = self.side_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

























class TensorProduct_protein_ScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=False, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(TensorProduct_protein_ScoreModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.num_conv_layers = num_conv_layers
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        # embedding layers


        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.atom_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.atom_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.ar_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
        self.atom_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        # convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            for _ in range(4): # 3 intra & 6 inter per each layer
                conv_layers.append(TensorProductConvLayer(**parameters))

        self.conv_layers = nn.ModuleList(conv_layers)




        self.side_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
        self.tor_bond_conv = TensorProductConvLayer(
            in_irreps=self.conv_layers[-1].out_irreps,
            sh_irreps=self.final_tp_tor.irreps_out,
            out_irreps=f'{ns}x0o + {ns}x0e',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.tor_final_layer = nn.Sequential(
                nn.Linear(2 * ns, ns, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False)
            )


    def forward(self, data):
        if not self.confidence_mode:
            tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['t']])
        else:
            tor_sigma = [data.complex_t[noise_type] for noise_type in ['t']]

        # build ligand graph
        cross_cutoff = (data.complex_t["t"] * 10 + 5).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance

        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build atom graph
        atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(data,cross_cutoff)
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

        # build cross graph
        ar_edge_index, ar_edge_attr, ar_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)



        for l in range(self.num_conv_layers):
            # ATOM UPDATES

            atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[atom_edge_index[0], :self.ns], atom_node_attr[atom_edge_index[1], :self.ns]], -1)
            atom_update = self.conv_layers[4*l](atom_node_attr, atom_edge_index, atom_edge_attr_, atom_edge_sh)


            ar_edge_attr_ = torch.cat([ar_edge_attr, atom_node_attr[ar_edge_index[0], :self.ns], rec_node_attr[ar_edge_index[1], :self.ns]],-1)
            ar_update = self.conv_layers[4*l+1](rec_node_attr, ar_edge_index, ar_edge_attr_, ar_edge_sh, out_nodes=atom_node_attr.shape[0])


            if l != self.num_conv_layers-1:  # last layer optimisation

                # LIGAND updates


                # RECEPTOR updates
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_edge_index[0], :self.ns], rec_node_attr[rec_edge_index[1], :self.ns]], -1)
                rec_update = self.conv_layers[4*l+2](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)


                ra_edge_attr_ = torch.cat([ar_edge_attr, rec_node_attr[ar_edge_index[1], :self.ns], atom_node_attr[ar_edge_index[0], :self.ns]], -1)
                ra_update = self.conv_layers[4*l+3](atom_node_attr, torch.flip(ar_edge_index, dims=[0]), ra_edge_attr_,
                                                    ar_edge_sh, out_nodes=rec_node_attr.shape[0])

            # padding original features and update features with residual updates
            atom_node_attr = F.pad(atom_node_attr, (0, atom_update.shape[-1] - rec_node_attr.shape[-1]))
            atom_node_attr = atom_node_attr + atom_update  + ar_update
            

            if l != self.num_conv_layers - 1:  # last layer optimisation

                rec_node_attr = F.pad(rec_node_attr, (0, rec_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_update + ra_update 

        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_side_bond_conv_graph(data)
        tor_bond_vec = data['atom'].pos[tor_bonds[1]] - data['atom'].pos[tor_bonds[0]]
        tor_bond_attr = atom_node_attr[tor_bonds[0]] + atom_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, atom_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)

        tor_pred = self.tor_bond_conv(atom_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                      out_nodes=data.tor_mask.sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['atom'].batch][data['atom', 'atom_contact', 'atom'].edge_index[0]][data.tor_mask]
        tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['atom'].x.device))

        return tor_pred


    def build_rec_conv_graph(self, data):
        # build the graph between receptor residues
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['t'])
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_atom_conv_graph(self, data,cross_cutoff):
        # build the graph between receptor atoms
        data['atom'].node_sigma_emb = self.timestep_emb_func(data['atom'].node_t['t'])
        node_attr = torch.cat([data['atom'].x, data['atom'].node_sigma_emb], 1)
        radius_edges = radius_graph(data['atom'].pos, self.lig_max_radius, data['atom'].batch)
        edge_index = torch.cat([data['atom', 'atom_contact', 'atom'].edge_index, radius_edges], 1).long()
        edge_index = torch.unique(edge_index.T, dim=0)
        edge_index = edge_index.T
        src, dst = edge_index
        edge_vec = data['atom'].pos[dst.long()] - data['atom'].pos[src.long()]

        edge_length_emb = self.atom_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['atom'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between receptor residues and receptor atoms
        ar_edge_index = data['atom', 'receptor'].edge_index
        ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['atom'].pos[ar_edge_index[0].long()]
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data['atom'].node_sigma_emb[ar_edge_index[0].long()]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')

        return ar_edge_index, ar_edge_attr, ar_edge_sh



    def build_side_bond_conv_graph(self, data):
        bonds = data['atom', 'atom_contact', 'atom'].edge_index[:, data.tor_mask].long()
        bond_pos = (data['atom'].pos[bonds[0]] + data['atom'].pos[bonds[1]]) / 2
        bond_batch = data['atom'].batch[bonds[0]]
        edge_index = radius(data['atom'].pos, bond_pos, self.lig_max_radius, batch_x=data['atom'].batch, batch_y=bond_batch)
        
        edge_vec = data['atom'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.atom_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = self.side_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh



