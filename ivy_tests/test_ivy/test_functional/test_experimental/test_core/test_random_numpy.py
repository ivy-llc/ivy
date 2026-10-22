"""Regression tests for NumPy-specific random behavior."""

import numpy as np
import pytest

import ivy
from ivy.functional.frontends.numpy import random as numpy_frontend_random


@pytest.fixture
def numpy_backend(backend_fw):
    if backend_fw != "numpy":
        pytest.skip("NumPy backend regression")

    original_state = np.random.get_state()
    ivy.set_backend("numpy")
    try:
        yield
    finally:
        ivy.unset_backend()
        np.random.set_state(original_state)


@pytest.mark.parametrize("distribution", ["beta", "gamma"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    ("alpha", "beta", "shape"),
    [
        pytest.param(1.5, 2.0, None, id="scalars"),
        pytest.param(1.5, np.array([0.5, 1.0, 2.0]), None, id="scalar-array"),
        pytest.param(np.array([1.5, 2.5, 3.5]), 2.0, None, id="array-scalar"),
        pytest.param(
            np.array(1.5), np.array([0.5, 1.0, 2.0]), None, id="zero-dimensional"
        ),
        pytest.param(
            np.array([[1.5], [2.5]]),
            np.array([0.5, 1.0, 2.0]),
            None,
            id="broadcast",
        ),
        pytest.param(
            np.array([1.5, 2.5, 3.5]),
            np.array([[0.5], [2.0]]),
            None,
            id="broadcast-reversed",
        ),
        pytest.param(
            np.array([[1.5], [2.5]]),
            np.array([0.5, 1.0, 2.0]),
            (4, 2, 3),
            id="explicit-expanded-shape",
        ),
        pytest.param(
            np.array([[1.5], [2.5]]),
            np.array([0.5, 1.0, 2.0]),
            [2, 3],
            id="explicit-list-shape",
        ),
    ],
)
def test_numpy_distribution_parameter_shapes(
    numpy_backend, distribution, dtype, alpha, beta, shape
):
    seed = 0
    result = getattr(ivy, distribution)(
        alpha, beta, shape=shape, dtype=dtype, seed=seed
    )
    reference_rng = np.random.RandomState(seed)
    expected = np.asarray(
        getattr(reference_rng, distribution)(alpha, beta, size=shape), dtype=dtype
    )

    actual = ivy.to_numpy(result)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("distribution", ["beta", "gamma"])
def test_numpy_distribution_rejects_incompatible_shape(numpy_backend, distribution):
    alpha = np.array([[1.5], [2.5]])
    beta = np.array([0.5, 1.0, 2.0])

    with pytest.raises(ValueError):
        getattr(ivy, distribution)(alpha, beta, shape=(2, 1), seed=0)


@pytest.mark.parametrize("distribution", ["beta", "gamma"])
def test_numpy_frontend_distribution_broadcasts_parameters(numpy_backend, distribution):
    alpha = np.array([[1.5], [2.5]])
    beta = np.array([0.5, 1.0, 2.0])
    seed = 7
    ivy.seed(seed_value=seed)

    result = getattr(numpy_frontend_random, distribution)(alpha, beta)
    reference_rng = np.random.RandomState(seed)
    expected = getattr(reference_rng, distribution)(alpha, beta)

    actual = ivy.to_numpy(result.ivy_array)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (2, 3)
    assert actual.dtype == expected.dtype
