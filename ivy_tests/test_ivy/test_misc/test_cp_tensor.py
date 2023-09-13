import ivy

import numpy as np
import pytest

# These tests have been adapetd from Tensorly
# https://github.com/tensorly/tensorly/blob/main/tensorly/tests/test_cp_tensor.py


@pytest.mark.parametrize(
    "shape, rank",
    [
        (
            (3, 4, 5),
            4,
        )
    ],
)
def test_cp_normalize(shape, rank):
    cp_tensor = ivy.random_cp(shape, rank)
    weights, factors = ivy.CPTensor.cp_normalize(cp_tensor)
    expected_norm = ivy.ones((rank,))
    for f in factors:
        norm = ivy.sqrt(ivy.sum(ivy.square(f), axis=0))
        assert np.allclose(norm, expected_norm)
    assert np.allclose(
        ivy.CPTensor.cp_to_tensor((weights, factors)),
        ivy.CPTensor.cp_to_tensor(cp_tensor),
    )


@pytest.mark.parametrize(
    "shape, rank",
    [
        (
            (3, 4, 5),
            4,
        )
    ],
)
def test_cp_flip_sign(shape, rank):
    cp_tensor = ivy.random_cp(shape, rank)
    weights, factors = ivy.CPTensor.cp_flip_sign(cp_tensor)

    assert ivy.all(ivy.mean(factors[1], axis=0) > 0)
    assert ivy.all(ivy.mean(factors[2], axis=0) > 0)
    assert cp_tensor.rank == cp_tensor.rank
    assert np.allclose(cp_tensor.weights, weights)
    assert np.allclose(
        ivy.CPTensor.cp_to_tensor((weights, factors)),
        ivy.CPTensor.cp_to_tensor(cp_tensor),
    )


@pytest.mark.parametrize(
    "true_shape, true_rank",
    [
        (
            (3, 4, 5),
            3,
        )
    ],
)
def test_validate_cp_tensor(true_shape, true_rank):
    cp_tensor = ivy.random_cp(true_shape, true_rank)
    (weights, factors) = ivy.CPTensor.cp_normalize(cp_tensor)

    # Check correct rank and shapes are returned
    shape, rank = ivy.CPTensor.validate_cp_tensor((weights, factors))
    np.testing.assert_equal(
        true_shape,
        shape,
        err_msg=f"Returned incorrect shape (got {shape}, expected {true_shape})",
    )
    np.testing.assert_equal(
        rank,
        true_rank,
        err_msg=f"Returned incorrect rank (got {rank}, expected {true_rank})",
    )

    # One of the factors has the wrong rank
    factors[0], copy = ivy.random_uniform(shape=(4, 4)), factors[0]
    with np.testing.assert_raises(ValueError):
        ivy.CPTensor.validate_cp_tensor((weights, factors))

    # Not the correct amount of weights
    factors[0] = copy
    wrong_weights = weights[1:]
    with np.testing.assert_raises(ValueError):
        ivy.CPTensor.validate_cp_tensor((wrong_weights, factors))

    # Not enough factors
    with np.testing.assert_raises(ValueError):
        ivy.CPTensor.validate_cp_tensor((weights[:1], factors[:1]))


@pytest.mark.parametrize(
    "shapeU1, shapeU2, shapeU3, shapeU4, true_res, columns, rows",
    [
        (
            (3, 3),
            (4, 3),
            (2, 3),
            (2, 3),
            [
                [
                    [[46754.0, 51524.0], [52748.0, 58130.0]],
                    [[59084.0, 65114.0], [66662.0, 73466.0]],
                    [[71414.0, 78704.0], [80576.0, 88802.0]],
                    [[83744.0, 92294.0], [94490.0, 104138.0]],
                ],
                [
                    [[113165.0, 124784.0], [127790.0, 140912.0]],
                    [[143522.0, 158264.0], [162080.0, 178730.0]],
                    [[173879.0, 191744.0], [196370.0, 216548.0]],
                    [[204236.0, 225224.0], [230660.0, 254366.0]],
                ],
                [
                    [[179576.0, 198044.0], [202832.0, 223694.0]],
                    [[227960.0, 251414.0], [257498.0, 283994.0]],
                    [[276344.0, 304784.0], [312164.0, 344294.0]],
                    [[324728.0, 358154.0], [366830.0, 404594.0]],
                ],
            ],
            4,
            (3, 4, 2),
        )
    ],
)
def test_cp_to_tensor(shapeU1, shapeU2, shapeU3, shapeU4, true_res, columns, rows):
    U1 = ivy.reshape(ivy.arange(1, 10, dtype=float), shapeU1)
    U2 = ivy.reshape(ivy.arange(10, 22, dtype=float), shapeU2)
    U3 = ivy.reshape(ivy.arange(22, 28, dtype=float), shapeU3)
    U4 = ivy.reshape(ivy.arange(28, 34, dtype=float), shapeU4)
    U = [ivy.array(t) for t in [U1, U2, U3, U4]]
    true_res = ivy.array(true_res)
    res = ivy.CPTensor.cp_to_tensor((ivy.ones(shape=(3,)), U))
    assert np.allclose(res, true_res)

    matrices = [
        ivy.arange(k * columns, dtype=float).reshape((k, columns)) for k in rows
    ]
    tensor = ivy.CPTensor.cp_to_tensor((ivy.ones(shape=(columns,)), matrices))
    for i in range(len(rows)):
        unfolded = ivy.unfold(tensor, mode=i)
        U_i = matrices.pop(i)
        reconstructed = ivy.matmul(
            U_i, ivy.permute_dims(ivy.khatri_rao(matrices), (1, 0))
        )
        assert np.allclose(reconstructed, unfolded)
        matrices.insert(i, U_i)


@pytest.mark.parametrize("shape, expected", [((2, 2), [[-2, -2], [6, 10]])])
def test_cp_to_tensor_with_weights(shape, expected):
    A = ivy.reshape(ivy.arange(1, 5, dtype=float), shape)
    B = ivy.reshape(ivy.arange(5, 9, dtype=float), shape)
    weights = ivy.array([2, -1], dtype=A.dtype)

    out = ivy.CPTensor.cp_to_tensor((weights, [A, B]))
    expected = ivy.array(expected)  # computed by hand
    assert np.allclose(out, expected)

    (weights, factors) = ivy.random_cp((5, 5, 5), 5, normalise_factors=True, full=False)
    true_res = ivy.matmul(
        ivy.matmul(factors[0], ivy.diag(weights)),
        ivy.permute_dims(ivy.khatri_rao(factors[1:]), (1, 0)),
    )
    true_res = ivy.fold(true_res, 0, (5, 5, 5))
    res = ivy.CPTensor.cp_to_tensor((weights, factors))
    assert np.allclose(true_res, res)


@pytest.mark.parametrize(
    "shapeU1, shapeU2, shapeU3, shapeU4", [((3, 3), (4, 3), (2, 3), (2, 3))]
)
def test_cp_to_unfolded(shapeU1, shapeU2, shapeU3, shapeU4):
    U1 = ivy.reshape(ivy.arange(1, 10, dtype=float), shapeU1)
    U2 = ivy.reshape(ivy.arange(10, 22, dtype=float), shapeU2)
    U3 = ivy.reshape(ivy.arange(22, 28, dtype=float), shapeU3)
    U4 = ivy.reshape(ivy.arange(28, 34, dtype=float), shapeU4)
    U = [ivy.array(t) for t in [U1, U2, U3, U4]]
    cp_tensor = ivy.CPTensor((ivy.ones((3,)), U))

    full_tensor = ivy.CPTensor.cp_to_tensor(cp_tensor)
    for mode in range(4):
        true_res = ivy.unfold(full_tensor, mode)
        res = ivy.CPTensor.cp_to_unfolded(cp_tensor, mode)
        assert np.allclose(
            true_res,
            res,
        )


@pytest.mark.parametrize(
    "shapeU1, shapeU2, shapeU3, shapeU4", [((3, 3), (4, 3), (2, 3), (2, 3))]
)
def test_cp_to_vec(shapeU1, shapeU2, shapeU3, shapeU4):
    """Test for cp_to_vec."""
    U1 = np.reshape(np.arange(1, 10, dtype=float), shapeU1)
    U2 = np.reshape(np.arange(10, 22, dtype=float), shapeU2)
    U3 = np.reshape(np.arange(22, 28, dtype=float), shapeU3)
    U4 = np.reshape(np.arange(28, 34, dtype=float), shapeU4)
    U = [ivy.array(t) for t in [U1, U2, U3, U4]]
    cp_tensor = ivy.CPTensor(
        (
            ivy.ones(
                (3),
            ),
            U,
        )
    )
    full_tensor = ivy.CPTensor.cp_to_tensor(cp_tensor)
    true_res = ivy.reshape(full_tensor, (-1))
    res = ivy.CPTensor.cp_to_vec(cp_tensor)
    assert np.allclose(true_res, res)


@pytest.mark.parametrize(
    "shape, rank",
    [
        (
            (5, 4, 6),
            3,
        )
    ],
)
def test_cp_mode_dot(shape, rank):
    cp_ten = ivy.random_cp(shape, rank, orthogonal=True, full=False)
    full_tensor = ivy.CPTensor.cp_to_tensor(cp_ten)
    # matrix for mode 1
    matrix = ivy.random_uniform(shape=(7, shape[1]))
    # vec for mode 2
    vec = ivy.random_uniform(shape=(shape[2]))

    # Test cp_mode_dot with matrix
    res = ivy.CPTensor.cp_mode_dot(cp_ten, matrix, mode=1, copy=True)
    # Note that if copy=True is not respected, factors will be changes
    # And the next test will fail
    res = ivy.CPTensor.cp_to_tensor(res)
    true_res = ivy.mode_dot(full_tensor, matrix, mode=1)
    assert np.allclose(true_res, res, atol=1e-3, rtol=1e-3)

    # Check that the data was indeed copied
    rec = ivy.CPTensor.cp_to_tensor(cp_ten)
    assert np.allclose(full_tensor, rec)

    # Test cp_mode_dot with vec
    res = ivy.CPTensor.cp_mode_dot(cp_ten, vec, mode=2, copy=True)
    res = ivy.CPTensor.cp_to_tensor(res)
    true_res = ivy.mode_dot(full_tensor, vec, mode=2)
    assert res.shape == true_res.shape
    assert np.allclose(true_res, res)


@pytest.mark.parametrize(
    "shape, rank, tol",
    [
        (
            (8, 5, 6, 4),
            25,
            10e-5,
        )
    ],
)
def test_cp_norm(shape, rank, tol):
    cp_tensor = ivy.random_cp(shape, rank, full=False, normalise_factors=True)
    rec = ivy.CPTensor.cp_to_tensor(cp_tensor)
    true_res = ivy.sqrt(ivy.sum(ivy.square(rec)))
    res = ivy.CPTensor.cp_norm(cp_tensor)
    assert ivy.abs(true_res - res) <= tol


@pytest.mark.parametrize("size", [4])
def test_validate_cp_rank(size):
    tensor_shape = tuple(ivy.randint(1, 100, shape=(size,)))
    n_param_tensor = ivy.prod(tensor_shape)

    # Rounding = floor
    rank = ivy.CPTensor.validate_cp_rank(tensor_shape, rank="same", rounding="floor")
    n_param = ivy.CPTensor.cp_n_param(tensor_shape, rank)
    assert n_param <= n_param_tensor

    # Rounding = ceil
    rank = ivy.CPTensor.validate_cp_rank(tensor_shape, rank="same", rounding="ceil")
    n_param = ivy.CPTensor.cp_n_param(tensor_shape, rank)
    assert n_param >= n_param_tensor


@pytest.mark.parametrize(
    "shape, rank",
    [
        (
            (8, 5, 6, 4),
            25,
        )
    ],
)
def test_cp_lstsq_grad(shape, rank):
    """Validate the gradient calculation between a CP and dense tensor."""
    cp_tensor = ivy.random_cp(shape, rank, normalise_factors=False)

    # If we're taking the gradient of comparison with self it should be 0
    cp_grad = ivy.CPTensor.cp_lstsq_grad(
        cp_tensor, ivy.CPTensor.cp_to_tensor(cp_tensor)
    )
    assert ivy.CPTensor.cp_norm(cp_grad) <= 10e-5

    # Check that we can solve for a direction of descent
    dense = ivy.random_cp(shape, rank, full=True, normalise_factors=False)
    cost_before = ivy.sqrt(
        ivy.sum(ivy.square(ivy.CPTensor.cp_to_tensor(cp_tensor) - dense))
    )

    cp_grad = ivy.CPTensor.cp_lstsq_grad(cp_tensor, dense)
    cp_new = ivy.CPTensor(cp_tensor)
    for ii in range(len(shape)):
        cp_new.factors[ii] = cp_tensor.factors[ii] - 1e-3 * cp_grad.factors[ii]

    cost_after = ivy.sqrt(
        ivy.sum(ivy.square(ivy.CPTensor.cp_to_tensor(cp_new) - dense))
    )
    assert cost_before > cost_after


@pytest.mark.parametrize(
    "shape, rank",
    [
        (
            (10, 10, 10, 4),
            5,
        )
    ],
)
def test_unfolding_dot_khatri_rao(shape, rank):
    tensor = ivy.random_uniform(shape=shape)
    weights, factors = ivy.random_cp(shape, rank, full=False, normalise_factors=True)

    for mode in range(4):
        # Version forming explicitely the khatri-rao product
        unfolded = ivy.unfold(tensor, mode)
        kr_factors = ivy.khatri_rao(factors, weights=weights, skip_matrix=mode)
        true_res = ivy.matmul(unfolded, kr_factors)

        # Efficient sparse-safe version
        res = ivy.CPTensor.unfolding_dot_khatri_rao(tensor, (weights, factors), mode)
        assert np.allclose(true_res, res)
