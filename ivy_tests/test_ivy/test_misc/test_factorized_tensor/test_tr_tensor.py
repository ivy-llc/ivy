import ivy

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("shape1", "shape2", "shape3"),
    [
        (
            (2, 4, 3),
            (3, 5, 2),
            (2, 6, 2),
        )
    ],
)
def test_tr_to_tensor(shape1, shape2, shape3):
    # Create ground truth TR factors
    factors = [
        ivy.random_uniform(shape=shape1),
        ivy.random_uniform(shape=shape2),
        ivy.random_uniform(shape=shape3),
    ]

    # Create tensor
    tensor = ivy.einsum("iaj,jbk,kci->abc", *factors)

    # Check that TR factors re-assemble to the original tensor
    assert np.allclose(tensor, ivy.TRTensor.tr_to_tensor(factors), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("rank1", "rank2"),
    [((2, 3, 4, 2), (2, 3, 4, 2, 3))],
)
def test_validate_tr_rank(rank1, rank2):
    tensor_shape = tuple(np.random.randint(1, 100, size=4))
    n_param_tensor = np.prod(tensor_shape)

    # Rounding = floor
    rank = ivy.TRTensor.validate_tr_rank(tensor_shape, rank="same", rounding="floor")
    n_param = ivy.TRTensor.tr_n_param(tensor_shape, rank)
    assert n_param <= n_param_tensor

    # Rounding = ceil
    rank = ivy.TRTensor.validate_tr_rank(tensor_shape, rank="same", rounding="ceil")
    n_param = ivy.TRTensor.tr_n_param(tensor_shape, rank)
    assert n_param >= n_param_tensor

    # Integer rank
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_rank(tensor_shape, rank=rank1)

    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_rank(tensor_shape, rank=rank2)


# These tests have been adapted from Tensorly
# https://github.com/tensorly/tensorly/blob/main/tensorly/tests/test_tr_tensor.py


@pytest.mark.parametrize(
    ("true_shape", "true_rank"),
    [
        (
            (6, 4, 5),
            (3, 2, 2, 3),
        )
    ],
)
def test_validate_tr_tensor(true_shape, true_rank):
    factors = ivy.random_tr(true_shape, true_rank).factors

    # Check correct rank and shapes are returned
    shape, rank = ivy.TRTensor.validate_tr_tensor(factors)
    np.testing.assert_equal(
        shape,
        true_shape,
        err_msg=f"Returned incorrect shape (got {shape}, expected {true_shape})",
    )
    np.testing.assert_equal(
        rank,
        true_rank,
        err_msg=f"Returned incorrect rank (got {rank}, expected {true_rank})",
    )

    # One of the factors has the wrong ndim
    factors[0] = ivy.random_uniform(shape=(4, 4))
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_tensor(factors)

    # Consecutive factors ranks don't match
    factors[0] = ivy.random_uniform(shape=(3, 6, 4))
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_tensor(factors)

    # Boundary conditions not respected
    factors[0] = ivy.random_uniform(shape=(2, 6, 2))
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_tensor(factors)
