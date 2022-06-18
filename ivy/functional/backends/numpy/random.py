"""Collection of Numpy random functions, wrapped to fit Ivy syntax and signature."""

# global
import numpy as np
from typing import Optional, Union, Tuple, Sequence

# localf


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype=None,
    *,
    device: str
) -> np.ndarray:
    return np.asarray(np.random.uniform(low, high, shape), dtype=dtype)


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    device: str,
) -> np.ndarray:
    return np.asarray(np.random.normal(mean, std, shape))


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[np.ndarray] = None,
    replace=True,
    *,
    device: str
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
    probs_flat = probs_flat / np.sum(probs_flat, -1, keepdims=True, dtype="float64")
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
    shape: Union[int, Sequence[int]],
    *,
    device: str
) -> np.ndarray:
    return np.random.randint(low, high, shape)


def seed(seed_value: int = 0) -> None:
    np.random.seed(seed_value)


def shuffle(x: np.ndarray) -> np.ndarray:
    return np.random.permutation(x)
