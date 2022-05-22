"""Collection of Numpy random functions, wrapped to fit Ivy syntax and signature."""

# global
import numpy as np
from typing import Optional, Union, Tuple

# local
import ivy


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    device: Optional[Union[ivy.Device, str]] = None,
) -> np.ndarray:
    return np.asarray(np.random.uniform(low, high, shape))


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    device: Optional[Union[ivy.Device, str]] = None,
) -> np.ndarray:
    return np.asarray(np.random.normal(mean, std, shape))


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[np.ndarray] = None,
    replace=True,
    device: Optional[Union[ivy.Device, str]] = None,
) -> np.ndarray:
    if probs is None:
        probs = (
            np.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = np.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / np.sum(probs_flat, -1, keepdims=True)
    probs_stack = np.split(probs_flat, probs_flat.shape[0])
    samples_stack = [
        np.random.choice(num_classes, num_samples, replace, p=prob[0])
        for prob in probs_stack
    ]
    samples_flat = np.stack(samples_stack)
    return np.asarray(np.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]))


def randint(
    low: int,
    high: int,
    shape: Union[int, Tuple[int, ...]],
    device: Optional[Union[ivy.Device, str]] = None,
) -> np.ndarray:
    return np.random.randint(low, high, shape)


seed = lambda seed_value=0: np.random.seed(seed_value)


def shuffle(x: np.ndarray) -> np.ndarray:
    return np.random.permutation(x)
