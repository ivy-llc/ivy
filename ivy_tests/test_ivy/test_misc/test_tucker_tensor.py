import ivy

import numpy as np
import pytest


# These tests have been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tests/test_tucker_tensor.py
@pytest.mark.parametrize("true_shape, true_rank", [((3, 4, 5), (3, 2, 4))])
def test_validate_tucker_tensor(true_shape, true_rank):
    core, factors = ivy.random_tucker(true_shape, true_rank)

    # Check shape and rank returned
    shape, rank = ivy.TuckerTensor.validate_tucker_tensor((core, factors))
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

    # One of the factors has the wrong rank
    factors[0], copy = ivy.random_uniform(shape=((4, 4))), factors[0]
    with np.testing.assert_raises(ValueError):
        ivy.TuckerTensor.validate_tucker_tensor((core, factors))

    # Not enough factors to match core
    factors[0] = copy
    with np.testing.assert_raises(ValueError):
        ivy.TuckerTensor.validate_tucker_tensor((core, factors[1:]))

    # Not enough factors
    with np.testing.assert_raises(ValueError):
        ivy.TuckerTensor.validate_tucker_tensor((core, factors[:1]))


@pytest.mark.parametrize(
    "X, ranks, true_res",
    [
        (
            [
                [[1.0, 13], [4, 16], [7, 19], [10, 22]],
                [[2, 14], [5, 17], [8, 20], [11, 23]],
                [[3, 15], [6, 18], [9, 21], [12, 24]],
            ],
            [2, 3, 4],
            [
                [
                    [390.0, 1518, 2646, 3774],
                    [1310, 4966, 8622, 12278],
                    [2230, 8414, 14598, 20782],
                ],
                [
                    [1524, 5892, 10260, 14628],
                    [5108, 19204, 33300, 47396],
                    [8692, 32516, 56340, 80164],
                ],
            ],
        )
    ],
)
def test_tucker_to_tensor(X, ranks, true_res):
    """Test for tucker_to_tensor."""
    X = ivy.array(X)
    U = [
        ivy.arange(R * s, dtype=ivy.float32).reshape((R, s))
        for (R, s) in zip(ranks, X.shape)
    ]
    true_res = ivy.array(true_res)
    res = ivy.TuckerTensor.tucker_to_tensor((X, U))
    assert np.allclose(true_res, res)
