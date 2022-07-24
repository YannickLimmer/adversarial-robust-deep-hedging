from typing import Union, Tuple
import numpy as np
import torch


def negative_standard_deviation(arr: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return - torch.std(arr, dim=dim, unbiased=True)


def mean_variance(arr: torch.Tensor, scale: np.float = 1, dim: int = 0) -> torch.Tensor:
    return torch.mean(arr, dim=dim) - 0.5 * scale * torch.var(arr, dim=dim, unbiased=True)


def entropy(arr: torch.Tensor, scale: float = 1, dim: int = 0) -> torch.Tensor:
    """
    Calculates the negative entropy of a given array, according to the formula

    .. math::
    - (\frac{1}{\\lambda}\\log(E[\\exp(-\\lambda (X-h))]) - h)

    which is invariant in h. This implies, that minimizing an entropy is not connected to the expectation of the array,
    since uniform shifts in values result in uniform shifts of the entropy function.

    Note that we use the negative of the entropy to continue with our convention that high values are yielding a higher
    'goodness', see for instance mean-variance.

    :param arr: The array, corresponds to X.
    :param scale: The risk-aversion, corresponds to h.
    :param dim: The axis the operation is performed on.
    :return: The entropy of the given array.
    """
    return - torch.log(torch.mean(torch.exp(- scale * arr), dim=dim)) / scale


def negative_standard_error(arr: torch.Tensor, dim: int = 0):
    return - torch.mean(arr ** 2, dim=dim)


def conditional_value_at_risk(
        arr: torch.Tensor,
        considered_fraction: np.float = 0.2,
        dim: int = 0,
) -> torch.Tensor:
    """
    Calculates the value at risk for a given array. The CVAR is the mean over a part of the array, where only the worst
    entries are considered. The number of worst samples considered is specified via a fraction.

    Further, the convention is supported that loss helper functions are interpreted as 'good' if they display high
    values. The function itself is not invariant w.r.t. the level it operates on, however, is invariant w.r.t. to scalar
    multiplication.

    :param arr: Array that the CVAR is calculated on, with minimal dimension of 1.
    :param considered_fraction: The portion of worst cases to be considered for taking the mean.
    :param dim: Axis along the CVAR is calculated.
    :return:
    """
    sorted_arr, _ = torch.sort(arr, dim=dim)
    considered_indices = create_slice(arr.shape, dim, considered_fraction)
    return torch.mean(sorted_arr[considered_indices], dim=dim)


def create_slice(shape: Tuple[int], axis: int, considered_fraction: np.float) -> Tuple:
    reduced_slice = np.s_[:int(considered_fraction * shape[axis]) + 1]
    return tuple((reduced_slice if ax == axis else np.s_[:]) for ax, _ in enumerate(shape))
