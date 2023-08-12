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


@pytest.mark.parametrize("shape, ranks", [((4, 3, 5, 2), (2, 2, 3, 4))])
def test_tucker_to_unfolded(shape, ranks):
    G = ivy.random_uniform(shape=shape)
    U = [ivy.random_uniform(shape=(ranks[i], G.shape[i])) for i in range(4)]
    full_tensor = ivy.TuckerTensor.tucker_to_tensor((G, U))
    for mode in range(4):
        assert np.allclose(
            ivy.TuckerTensor.tucker_to_unfolded((G, U), mode),
            ivy.unfold(full_tensor, mode),
        )
        assert np.allclose(
            ivy.TuckerTensor.tucker_to_unfolded((G, U), mode),
            ivy.dot(
                ivy.dot(U[mode], ivy.unfold(G, mode)),
                ivy.permute_dims(ivy.kronecker(U, skip_matrix=mode), (1, 0)),
            ),
        )


@pytest.mark.parametrize("shape, ranks", [((4, 3, 5, 2), (2, 2, 3, 4))])
def test_tucker_to_vec(shape, ranks):
    G = ivy.random_uniform(shape=shape)
    ranks = [2, 2, 3, 4]
    U = [ivy.random_uniform(shape=(ranks[i], G.shape[i])) for i in range(4)]
    vec = ivy.reshape(ivy.TuckerTensor.tucker_to_tensor((G, U)), -1)
    assert np.allclose(ivy.TuckerTensor.tucker_to_vec((G, U)), vec)
    assert np.allclose(
        ivy.TuckerTensor.tucker_to_vec((G, U)),
        ivy.dot(ivy.kronecker(U), ivy.reshape(G, -1)),
    )


@pytest.mark.parametrize("shape, ranks", [((5, 4, 6), (3, 2, 3))])
def test_tucker_mode_dot(shape, ranks):
    tucker_ten = ivy.random_tucker(shape, ranks, full=False)
    full_tensor = ivy.TuckerTensor.tucker_to_tensor(tucker_ten)
    # matrix for mode 1
    matrix = ivy.random_uniform(shape=(7, shape[1]))
    # vec for mode 2
    vec = ivy.random_uniform(shape=(shape[2]))

    # Test tucker_mode_dot with matrix
    res = ivy.TuckerTensor.tucker_mode_dot(tucker_ten, matrix, mode=1, copy=True)
    # Note that if copy=True is not respected, factors will be changes
    # And the next test will fail
    res = ivy.TuckerTensor.tucker_to_tensor(res)
    true_res = ivy.mode_dot(full_tensor, matrix, mode=1)
    assert np.allclose(true_res, res)

    # Check that the data was indeed copied
    rec = ivy.TuckerTensor.tucker_to_tensor(tucker_ten)
    assert np.allclose(full_tensor, rec)

    # Test tucker_mode_dot with vec
    res = ivy.TuckerTensor.tucker_mode_dot(tucker_ten, vec, mode=2, copy=True)
    res = ivy.TuckerTensor.tucker_to_tensor(res)
    true_res = ivy.mode_dot(full_tensor, vec, mode=2)
    assert np.allclose(res.shape, true_res.shape)
    assert np.allclose(true_res, res)
