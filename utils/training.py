import copy

import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from torch_geometric.data import Dataset, HeteroData
from utils import so3, torus
# import torus
from utils.sampling import randomize_position, sampling
import torch
from utils.diffusion_utils import get_t_schedule
from utils.utils import kabsch_rmsd



class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def loss_function(side_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, apply_mean=True, no_torsion=False):
    side_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['t']])
    mean_dims = (0, 1) if apply_mean else 1

    side_sigma = torch.from_numpy(
            np.concatenate([d.side_sigma for d in data] if device.type == 'cuda' else data.side_sigma))
    side_score = torch.cat([d.side_score for d in data], dim=0) if device.type == 'cuda' else data.side_score
    side_score_norm2 = torch.tensor(torus.score_norm(side_sigma.cpu().numpy())).float()
    side_loss = ((side_pred.cpu() - side_score) ** 2 / side_score_norm2)
    side_base_loss = ((side_score ** 2 / side_score_norm2)).detach()
    if apply_mean:
            side_loss, side_base_loss = side_loss.mean() * torch.ones(1, dtype=torch.float), side_base_loss.mean() * torch.ones(1, dtype=torch.float)
    else:
        print("please use apply_mean")
    loss = side_loss
    return loss, side_loss.detach(), side_base_loss

class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths):
    model.train()
    meter = AverageMeter(['loss', 'side_loss', 'side_base_loss'])

    for data in tqdm(loader, total=len(loader)):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            side_pred = model(data)
            loss, side_loss, side_base_loss = \
                loss_fn(side_pred, data=data, t_to_sigma=t_to_sigma, device=device)
            loss.backward()
            optimizer.step()
            ema_weigths.update(model.parameters())
            meter.add([loss.cpu().detach(), side_loss, side_base_loss])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'side_loss', 'side_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'side_loss', 'side_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                side_pred = model(data)

            loss, side_loss, side_base_loss = \
                loss_fn(side_pred, data=data, t_to_sigma=t_to_sigma, device=device)

            meter.add([loss.cpu().detach(), side_loss, side_base_loss])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch(model, complex_graphs, device, t_to_sigma, args):
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    side_schedule = t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds = []

    for orig_complex_graph in tqdm(loader):
        data_list = [copy.deepcopy(orig_complex_graph)]
        randomize_position(data_list, args.no_torsion, False)

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type=='cuda' else model,
                                                         inference_steps=args.inference_steps,side_schedule=side_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        if failed_convergence_counter > 5: continue
        if args.no_torsion:
            orig_complex_graph['atom'].orig_pos = orig_complex_graph['atom'].pos.cpu().numpy()

        filterHs = torch.not_equal(predictions_list[0]['atom'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['atom'].orig_pos, list):
            orig_complex_graph['atom'].orig_pos = orig_complex_graph['atom'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['atom'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        orig_ligand_pos = np.expand_dims(orig_complex_graph['atom'].orig_pos[filterHs], axis=0)
        N = len(predictions_list)
        rotated_atom_num = orig_complex_graph.rotated_atom_num
        rmsd = kabsch_rmsd(orig_ligand_pos, ligand_pos, N, rotated_atom_num)
        rmsds.append(rmsd)

    rmsds = np.array(rmsds)
    mean_rmsd = np.mean(rmsds)
    losses = {'rmsd':mean_rmsd,
            'rmsds_lt1': (100 * (rmsds < 1).sum() / len(rmsds)),
            'rmsds_lt1.25': (100 * (rmsds < 1.25).sum() / len(rmsds))}
    return losses
