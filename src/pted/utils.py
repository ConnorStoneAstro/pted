import numpy as np
from scipy.spatial.distance import cdist
import torch

__all__ = ["_pted_numpy", "_pted_torch"]


def _energy_distance_precompute(D, ix, iy):
    nx = len(ix)
    ny = len(iy)
    Exx = (D[ix.reshape(nx, 1), ix.reshape(1, nx)]).sum() / nx**2
    Eyy = (D[iy.reshape(ny, 1), iy.reshape(1, ny)]).sum() / ny**2
    Exy = (D[ix.reshape(nx, 1), iy.reshape(1, ny)]).sum() / (nx * ny)
    return 2 * Exy - Exx - Eyy


def _pted_numpy(x, y, permutations=100, metric="euclidean", return_all=False):
    z = np.concatenate((x, y), axis=0)
    dmatrix = cdist(z, z, metric=metric)
    nx = len(x)
    I = np.arange(len(z))

    test_stat = _energy_distance_precompute(dmatrix, I[:nx], I[nx:])
    permute_stats = []
    for _ in range(permutations):
        np.random.shuffle(I)
        permute_stats.append(_energy_distance_precompute(dmatrix, I[:nx], I[nx:]))
    if return_all:
        return test_stat, permute_stats
    # Compute p-value
    return np.mean(permute_stats > test_stat)


@torch.no_grad()
def _pted_torch(x, y, permutations=100, metric="euclidean", return_all=False):
    z = torch.cat((x, y), dim=0)
    if metric == "euclidean":
        metric = 2.0
    dmatrix = torch.cdist(z, z, metric=metric)
    nx = len(x)
    I = torch.arange(len(z))

    test_stat = _energy_distance_precompute(dmatrix, I[:nx], I[nx:]).item()
    permute_stats = []
    for _ in range(permutations):
        I = I[torch.randperm(len(I))]
        permute_stats.append(_energy_distance_precompute(dmatrix, I[:nx], I[nx:]).item())
    if return_all:
        return test_stat, permute_stats
    # Compute p-value
    return torch.mean(permute_stats > test_stat).item()
