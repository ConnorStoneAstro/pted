import numpy as np
from scipy.spatial.distance import cdist
import torch

__all__ = ["_pted_numpy", "_pted_torch"]


def _energy_distance_precompute(D, nx, ny):
    Exx = D[:nx, :nx].sum() / nx**2
    Eyy = D[nx:, nx:].sum() / ny**2
    Exy = D[:nx, nx:].sum() / (nx * ny)
    return 2 * Exy - Exx - Eyy


def _pted_numpy(x, y, permutations=100, metric="euclidean", return_all=False):
    z = np.concatenate((x, y), axis=0)
    assert np.all(np.isfinite(z)), "Input contains NaN or Inf!"
    dmatrix = cdist(z, z, metric=metric)
    assert np.all(
        np.isfinite(dmatrix)
    ), "Distance matrix contains NaN or Inf! Consider using a different metric or normalizing values to be more stable (i.e. z-score norm)."
    nx = len(x)
    ny = len(y)

    test_stat = _energy_distance_precompute(dmatrix, nx, ny)
    permute_stats = []
    for _ in range(permutations):
        I = np.random.permutation(len(z))
        dmatrix = dmatrix[I][:, I]
        permute_stats.append(_energy_distance_precompute(dmatrix, nx, ny))
    if return_all:
        return test_stat, permute_stats
    # Compute p-value
    return np.mean(np.array(permute_stats) > test_stat)


@torch.no_grad()
def _pted_torch(x, y, permutations=100, metric="euclidean", return_all=False):
    z = torch.cat((x, y), dim=0)
    assert torch.all(torch.isfinite(z)), "Input contains NaN or Inf!"
    if metric == "euclidean":
        metric = 2.0
    dmatrix = torch.cdist(z, z, p=metric)
    assert torch.all(
        torch.isfinite(dmatrix)
    ), "Distance matrix contains NaN or Inf! Consider using a different metric or normalizing values to be more stable (i.e. z-score norm)."
    nx = len(x)
    ny = len(y)

    test_stat = _energy_distance_precompute(dmatrix, nx, ny).item()
    permute_stats = []
    for _ in range(permutations):
        I = torch.randperm(len(z))
        dmatrix = dmatrix[I][:, I]
        permute_stats.append(_energy_distance_precompute(dmatrix, nx, ny).item())
    if return_all:
        return test_stat, permute_stats
    # Compute p-value
    return np.mean(np.array(permute_stats) > test_stat)
