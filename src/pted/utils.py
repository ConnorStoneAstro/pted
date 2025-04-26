import numpy as np
from scipy.spatial.distance import cdist
import torch

__all__ = ["_pted_numpy", "_pted_chunk_numpy", "_pted_torch", "_pted_chunk_torch"]


def _energy_distance_precompute(D, nx, ny):
    Exx = D[:nx, :nx].sum() / nx**2
    Eyy = D[nx:, nx:].sum() / ny**2
    Exy = D[:nx, nx:].sum() / (nx * ny)
    return 2 * Exy - Exx - Eyy


def _energy_distance_estimate(x, y, chunk_size, chunk_iter, metric="euclidean"):
    is_torch = isinstance(x, torch.Tensor)

    E_est = []
    for _ in range(chunk_iter):
        # Randomly sample a chunk of data
        idx = np.random.choice(len(x), size=min(len(x), chunk_size), replace=False)
        if is_torch:
            idx = torch.tensor(idx, device=x.device)
        x_chunk = x[idx]
        idy = np.random.choice(len(y), size=min(len(y), chunk_size), replace=False)
        if is_torch:
            idy = torch.tensor(idy, device=y.device)
        y_chunk = y[idy]

        # Compute the distance matrix
        if is_torch:
            z_chunk = torch.cat((x_chunk, y_chunk), dim=0)
        else:
            z_chunk = np.concatenate((x_chunk, y_chunk), axis=0)
        dmatrix = cdist(z_chunk, z_chunk, metric=metric)

        # Compute the energy distance
        E_est.append(_energy_distance_precompute(dmatrix, len(x_chunk), len(y_chunk)))
        if is_torch:
            E_est[-1] = E_est[-1].item()
    return np.mean(E_est)


def _pted_chunk_numpy(x, y, permutations=100, metric="euclidean", chunk_size=100, chunk_iter=10):
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)), "Input contains NaN or Inf!"
    nx = len(x)

    test_stat = _energy_distance_estimate(x, y, chunk_size, chunk_iter, metric=metric)
    permute_stats = []
    for _ in range(permutations):
        z = np.concatenate((x, y), axis=0)
        z = z[np.random.permutation(len(z))]
        x, y = z[:nx], z[nx:]
        permute_stats.append(_energy_distance_estimate(x, y, chunk_size, chunk_iter, metric=metric))
    return test_stat, permute_stats


def _pted_chunk_torch(x, y, permutations=100, metric="euclidean", chunk_size=100, chunk_iter=10):
    assert torch.all(torch.isfinite(x)) and torch.all(
        torch.isfinite(y)
    ), "Input contains NaN or Inf!"
    nx = len(x)

    test_stat = _energy_distance_estimate(x, y, chunk_size, chunk_iter, metric=metric)
    permute_stats = []
    for _ in range(permutations):
        z = torch.cat((x, y), dim=0)
        z = z[torch.randperm(len(z))]
        x, y = z[:nx], z[nx:]
        permute_stats.append(_energy_distance_estimate(x, y, chunk_size, chunk_iter, metric=metric))
    return test_stat, permute_stats


def _pted_numpy(x, y, permutations=100, metric="euclidean"):
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
    return test_stat, permute_stats


@torch.no_grad()
def _pted_torch(x, y, permutations=100, metric="euclidean"):
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
    return test_stat, permute_stats
