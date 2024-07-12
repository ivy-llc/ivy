import ivy

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("weights", "factors", "projections", "true_res"),
    [
        (
            (2, 3),
            [[[1, 1], [1, 0]], [[2, 1], [1, 2]], [[1, 1], [1, 0], [1, 0]]],
            [[[1, 0], [0, 1]], [[1, 0], [0, 0], [0, -1]]],
            [[[7, 4, 4], [8, 2, 2]], [[4, 4, 4], [0, 0, 0], [-2, -2, -2]]],
        )
    ],
)
def test_apply_parafac2_projections(weights, factors, projections, true_res):
    weights = ivy.array(weights)
    factors = [ivy.array(f) for f in factors]
    projections = [ivy.array(p) for p in projections]
    true_res = [ivy.array(t) for t in true_res]
    new_weights, projected_factors = ivy.Parafac2Tensor.apply_parafac2_projections(
        (weights, factors, projections)
    )
    np.allclose(new_weights, weights)
    for i, Bi in enumerate(projected_factors[1]):
        np.allclose(ivy.dot(projections[i], factors[1]), Bi)


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            [(4, 5)] * 3,
            2,
        )
    ],
)
def test_parafac2_normalise(shape, rank):
    parafac2_tensor = ivy.random_parafac2(shape, rank)

    normalised_parafac2_tensor = ivy.Parafac2Tensor.parafac2_normalise(
        parafac2_tensor
    )  # , copy=copy)
    expected_norm = ivy.ones((rank,))
    for f in normalised_parafac2_tensor[1]:
        norm = ivy.sqrt(ivy.sum(ivy.square(f), axis=0))
        assert np.allclose(norm, expected_norm)
    assert np.allclose(
        ivy.Parafac2Tensor.parafac2_to_tensor(parafac2_tensor),
        ivy.Parafac2Tensor.parafac2_to_tensor(normalised_parafac2_tensor),
    )


@pytest.mark.parametrize(
    ("weights", "factors", "projections", "true_res"),
    [
        (
            (2, 3),
            [[[1, 1], [1, 0]], [[2, 1], [1, 2]], [[1, 1], [1, 0], [1, 0]]],
            [[[1, 0], [0, 1]], [[1, 0], [0, 0], [0, -1]]],
            [[[7, 4, 4], [8, 2, 2]], [[4, 4, 4], [0, 0, 0], [-2, -2, -2]]],
        )
    ],
)
def test_parafac2_to_slices(weights, factors, projections, true_res):
    weights = ivy.array(weights)
    factors = [ivy.array(f) for f in factors]
    projections = [ivy.array(p) for p in projections]
    true_res = [ivy.array(t) for t in true_res]
    for i, true_slice in enumerate(true_res):
        assert np.allclose(
            ivy.Parafac2Tensor.parafac2_to_slice((weights, factors, projections), i),
            true_slice,
        )

    for true_slice, est_slice in zip(
        true_res, ivy.Parafac2Tensor.parafac2_to_slices((weights, factors, projections))
    ):
        np.allclose(true_slice, est_slice)


@pytest.mark.parametrize(
    ("weights", "factors", "projections", "true_res"),
    [
        (
            (2, 3),
            [[[1, 1], [1, 0]], [[2, 1], [1, 2]], [[1, 1], [1, 0], [1, 0]]],
            [[[0, 0], [1, 0], [0, 1]], [[1, 0], [0, 0], [0, -1]]],
            [[[0, 0, 0], [7, 4, 4], [8, 2, 2]], [[4, 4, 4], [0, 0, 0], [-2, -2, -2]]],
        )
    ],
)
def test_parafac2_to_tensor(weights, factors, projections, true_res):
    weights = ivy.array(weights)
    factors = [ivy.array(f) for f in factors]
    projections = [ivy.array(p) for p in projections]
    true_res = ivy.array(true_res)
    res = ivy.Parafac2Tensor.parafac2_to_tensor((weights, factors, projections))
    assert np.allclose(res, true_res)


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            [(4, 5)] * 3,
            2,
        )
    ],
)
def test_parafac2_to_unfolded(shape, rank):
    pf2_tensor = ivy.random_parafac2(shape, rank)
    full_tensor = ivy.Parafac2Tensor.parafac2_to_tensor(pf2_tensor)
    for mode in range(ivy.get_num_dims(full_tensor)):
        assert np.allclose(
            ivy.Parafac2Tensor.parafac2_to_unfolded(pf2_tensor, mode),
            ivy.unfold(full_tensor, mode),
        )


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            [(4, 5)] * 3,
            2,
        )
    ],
)
def test_parafac2_to_vec(shape, rank):
    pf2_tensor = ivy.random_parafac2(shape, rank)
    full_tensor = ivy.Parafac2Tensor.parafac2_to_tensor(pf2_tensor)
    np.allclose(
        ivy.Parafac2Tensor.parafac2_to_vec(pf2_tensor),
        ivy.reshape(full_tensor, (-1)),
    )


@pytest.mark.parametrize(
    ("true_shape", "true_rank"),
    [
        (
            [(4, 5)] * 3,
            2,
        )
    ],
)
def test_validate_parafac2_tensor(true_shape, true_rank):
    weights, factors, projections = ivy.random_parafac2(true_shape, true_rank)

    # Check shape and rank returned
    shape, rank = ivy.Parafac2Tensor.validate_parafac2_tensor(
        (weights, factors, projections)
    )
    np.testing.assert_equal(
        true_shape,
        shape,
        err_msg=f"Returned incorrect shape (got {shape}, expected {true_shape})",
    )
    np.testing.assert_equal(
        true_rank,
        rank,
        err_msg=f"Returned incorrect rank (got {rank}, expected {true_rank})",
    )

    # One of the factors has the wrong rank
    for mode in range(3):
        false_shape = (ivy.shape(factors[mode])[0], true_rank + 1)
        factors[mode], copy = ivy.random_uniform(shape=false_shape), factors[mode]
        with np.testing.assert_raises(ValueError):
            ivy.Parafac2Tensor.validate_parafac2_tensor((weights, factors, projections))

        factors[mode] = copy

    # Not three factor matrices
    with np.testing.assert_raises(ValueError):
        ivy.Parafac2Tensor.validate_parafac2_tensor((weights, factors[1:], projections))

    # Not enough projections
    with np.testing.assert_raises(ValueError):
        ivy.Parafac2Tensor.validate_parafac2_tensor((weights, factors, projections[1:]))

    # Wrong number of weights
    with np.testing.assert_raises(ValueError):
        ivy.Parafac2Tensor.validate_parafac2_tensor((weights[1:], factors, projections))

    # The projections aren't orthogonal
    false_projections = [ivy.random_uniform(shape=ivy.shape(P)) for P in projections]
    with np.testing.assert_raises(ValueError):
        ivy.Parafac2Tensor.validate_parafac2_tensor(
            (weights, factors, false_projections)
        )
