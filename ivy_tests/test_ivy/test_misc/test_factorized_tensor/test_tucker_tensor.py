import ivy

import numpy as np
import pytest


@pytest.mark.parametrize(("shape", "rank"), [((5, 4, 6), (3, 2, 3))])
def test_n_param_tucker(shape, rank):
    tucker_tensor = ivy.random_tucker(shape, rank)
    true_n_param = ivy.prod(ivy.shape(tucker_tensor[0])) + ivy.sum(
        [ivy.prod(ivy.shape(f)) for f in tucker_tensor[1]]
    )
    n_param = tucker_tensor.n_param
    assert np.allclose(n_param, true_n_param)


@pytest.mark.parametrize(("shape", "rank"), [((3, 4, 5), 4)])
def test_tucker_copy(shape, rank):
    tucker_tensor = ivy.random_tucker(shape, rank)
    core, factors = tucker_tensor
    core_normalized, factors_normalized = ivy.TuckerTensor.tucker_normalize(
        tucker_tensor.tucker_copy()
    )
    # Check that modifying copy tensor doesn't change the original tensor
    assert np.allclose(
        ivy.TuckerTensor.tucker_to_tensor((core, factors)),
        ivy.TuckerTensor.tucker_to_tensor(tucker_tensor),
    )


@pytest.mark.parametrize(("shape", "ranks"), [((5, 4, 6), (3, 2, 3))])
def test_tucker_mode_dot(shape, ranks):
    tucker_ten = ivy.random_tucker(shape, ranks, full=False)
    full_tensor = ivy.TuckerTensor.tucker_to_tensor(tucker_ten)
    # matrix for mode 1
    matrix = ivy.random_uniform(shape=(7, shape[1]))
    # vec for mode 2
    vec = ivy.random_uniform(shape=shape[2])

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


@pytest.mark.parametrize(("shape", "rank"), [((3, 4, 5), (3, 2, 4))])
def test_tucker_normalize(shape, rank):
    tucker_ten = ivy.random_tucker(shape, rank)
    core, factors = ivy.TuckerTensor.tucker_normalize(tucker_ten)
    for i in range(len(factors)):
        norm = ivy.sqrt(ivy.sum(ivy.abs(factors[i]) ** 2, axis=0))
        assert np.allclose(norm, ivy.ones(rank[i]))
    assert np.allclose(
        ivy.TuckerTensor.tucker_to_tensor((core, factors)),
        ivy.TuckerTensor.tucker_to_tensor(tucker_ten),
    )


@pytest.mark.parametrize(
    ("X", "ranks", "true_res"),
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


@pytest.mark.parametrize(("shape", "ranks"), [((4, 3, 5, 2), (2, 2, 3, 4))])
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


@pytest.mark.parametrize(("shape", "ranks"), [((4, 3, 5, 2), (2, 2, 3, 4))])
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


@pytest.mark.parametrize("tol", [(0.01)])
def test_validate_tucker_rank(tol):
    tensor_shape = tuple(ivy.randint(1, 100, shape=(5,)))
    n_param_tensor = ivy.prod(tensor_shape)

    # Rounding = floor
    rank = ivy.TuckerTensor.validate_tucker_rank(
        tensor_shape, rank="same", rounding="floor"
    )
    n_param = ivy.TuckerTensor.tucker_n_param(tensor_shape, rank)
    assert n_param * (1 - tol) <= n_param_tensor

    # Rounding = ceil
    rank = ivy.TuckerTensor.validate_tucker_rank(
        tensor_shape, rank="same", rounding="ceil"
    )
    n_param = ivy.TuckerTensor.tucker_n_param(tensor_shape, rank)
    assert n_param >= n_param_tensor * (1 - tol)

    # With fixed modes
    fixed_modes = [1, 4]
    tensor_shape = [
        s**2 if i in fixed_modes else s
        for (i, s) in enumerate(ivy.randint(2, 10, shape=(5,)))
    ]
    n_param_tensor = ivy.prod(tensor_shape)
    # Floor
    rank = ivy.TuckerTensor.validate_tucker_rank(
        tensor_shape, rank=0.5, fixed_modes=fixed_modes, rounding="floor"
    )
    n_param = ivy.TuckerTensor.tucker_n_param(tensor_shape, rank)
    for mode in fixed_modes:
        assert rank[mode] == tensor_shape[mode]
    assert n_param * (1 - tol) <= n_param_tensor * 0.5
    # Ceil
    fixed_modes = [0, 2]
    tensor_shape = [
        s**2 if i in fixed_modes else s
        for (i, s) in enumerate(ivy.randint(2, 10, shape=(5,)))
    ]
    n_param_tensor = ivy.prod(tensor_shape)
    rank = ivy.TuckerTensor.validate_tucker_rank(
        tensor_shape, rank=0.5, fixed_modes=fixed_modes, rounding="ceil"
    )
    n_param = ivy.TuckerTensor.tucker_n_param(tensor_shape, rank)
    for mode in fixed_modes:
        assert rank[mode] == tensor_shape[mode]
    assert n_param >= n_param_tensor * 0.5 * (1 - tol)


# These tests have been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tests/test_tucker_tensor.py
@pytest.mark.parametrize(("true_shape", "true_rank"), [((3, 4, 5), (3, 2, 4))])
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
