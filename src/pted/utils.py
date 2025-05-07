from typing import Optional, Union

import numpy as np
from scipy.spatial.distance import cdist
import torch

__all__ = ["_pted_numpy", "_pted_chunk_numpy", "_pted_torch", "_pted_chunk_torch"]


def _energy_distance_precompute(
    D: Union[np.ndarray, torch.Tensor], nx: int, ny: int
) -> Union[float, torch.Tensor]:
    Exx = D[:nx, :nx].sum() / nx**2
    Eyy = D[nx:, nx:].sum() / ny**2
    Exy = D[:nx, nx:].sum() / (nx * ny)
    return 2 * Exy - Exx - Eyy


def _energy_distance_numpy(x: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> float:
    nx = len(x)
    ny = len(y)
    z = np.concatenate((x, y), axis=0)
    D = cdist(z, z, metric=metric)
    return _energy_distance_precompute(D, nx, ny)


def _energy_distance_torch(
    x: torch.Tensor, y: torch.Tensor, metric: Union[str, float] = "euclidean"
) -> float:
    nx = len(x)
    ny = len(y)
    z = torch.cat((x, y), dim=0)
    if metric == "euclidean":
        metric = 2.0
    D = torch.cdist(z, z, p=metric)
    return _energy_distance_precompute(D, nx, ny).item()


def _energy_distance_estimate_numpy(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
    chunk_iter: int,
    metric: Union[str, float] = "euclidean",
) -> float:

    E_est = []
    for _ in range(chunk_iter):
        # Randomly sample a chunk of data
        idx = np.random.choice(len(x), size=min(len(x), chunk_size), replace=False)
        x_chunk = x[idx]
        idy = np.random.choice(len(y), size=min(len(y), chunk_size), replace=False)
        y_chunk = y[idy]

        # Compute the energy distance
        E_est.append(_energy_distance_numpy(x_chunk, y_chunk, metric=metric))
    return np.mean(E_est)


def _energy_distance_estimate_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int,
    chunk_iter: int,
    metric: Union[str, float] = "euclidean",
) -> float:

    E_est = []
    for _ in range(chunk_iter):
        # Randomly sample a chunk of data
        idx = np.random.choice(len(x), size=min(len(x), chunk_size), replace=False)
        x_chunk = x[torch.tensor(idx)]
        idy = np.random.choice(len(y), size=min(len(y), chunk_size), replace=False)
        y_chunk = y[torch.tensor(idy)]

        # Compute the energy distance
        E_est.append(_energy_distance_torch(x_chunk, y_chunk, metric=metric))
    return np.mean(E_est)


def _pted_chunk_numpy(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 100,
    metric: str = "euclidean",
    chunk_size: int = 100,
    chunk_iter: int = 10,
) -> tuple[float, list[float]]:
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)), "Input contains NaN or Inf!"
    nx = len(x)

    test_stat = _energy_distance_estimate_numpy(x, y, chunk_size, chunk_iter, metric=metric)
    permute_stats = []
    for _ in range(permutations):
        z = np.concatenate((x, y), axis=0)
        z = z[np.random.permutation(len(z))]
        x, y = z[:nx], z[nx:]
        permute_stats.append(
            _energy_distance_estimate_numpy(x, y, chunk_size, chunk_iter, metric=metric)
        )
    return test_stat, permute_stats


def _pted_chunk_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    permutations: int = 100,
    metric: Union[str, float] = "euclidean",
    chunk_size: int = 100,
    chunk_iter: int = 10,
) -> tuple[float, list[float]]:
    assert torch.all(torch.isfinite(x)) and torch.all(
        torch.isfinite(y)
    ), "Input contains NaN or Inf!"
    nx = len(x)

    test_stat = _energy_distance_estimate_torch(x, y, chunk_size, chunk_iter, metric=metric)
    permute_stats = []
    for _ in range(permutations):
        z = torch.cat((x, y), dim=0)
        z = z[torch.randperm(len(z))]
        x, y = z[:nx], z[nx:]
        permute_stats.append(
            _energy_distance_estimate_torch(x, y, chunk_size, chunk_iter, metric=metric)
        )
    return test_stat, permute_stats


def _pted_numpy(
    x: np.ndarray, y: np.ndarray, permutations: int = 100, metric: str = "euclidean"
) -> tuple[float, list[float]]:
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
def _pted_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    permutations: int = 100,
    metric: Union[str, float] = "euclidean",
) -> tuple[float, list[float]]:
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
