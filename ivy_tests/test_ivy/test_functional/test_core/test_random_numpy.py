"""Regression tests for NumPy-specific random behavior."""

import numpy as np
import pytest

import ivy


@pytest.mark.parametrize("seed", [0, 7, None])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "probs",
    [
        pytest.param(None, id="uniform-batch"),
        pytest.param([0.0625, 0.125, 0.1875, 0.25, 0.375], id="weighted-vector"),
        pytest.param(
            [
                [0.0625, 0.125, 0.1875, 0.25, 0.375],
                [0.375, 0.25, 0.1875, 0.125, 0.0625],
            ],
            id="weighted-batch",
        ),
    ],
)
def test_numpy_multinomial_seed(backend_fw, seed, replace, probs):
    if backend_fw != "numpy":
        pytest.skip("NumPy backend regression")

    probabilities = None if probs is None else np.asarray(probs)
    reference_probs = np.full((2, 5), 0.2) if probs is None else probabilities
    num_samples = 12 if replace else 4
    original_state = np.random.get_state()
    ivy.set_backend("numpy")
    try:
        for surrounding_seed in (11, 99):
            # Explicit seeds must override unrelated previous random draws.
            np.random.seed(surrounding_seed)
            np.random.random(9)
            result = ivy.multinomial(
                5,
                num_samples,
                batch_size=2,
                probs=probabilities,
                replace=replace,
                seed=seed,
            )

            reference_rng = np.random.RandomState(
                surrounding_seed if seed is None else seed
            )
            if seed is None:
                # Unseeded calls must continue using the existing RNG state.
                reference_rng.random_sample(9)
            expected = np.stack(
                [
                    reference_rng.choice(5, num_samples, replace=replace, p=row)
                    for row in np.atleast_2d(reference_probs)
                ]
            )
            if reference_probs.ndim == 1:
                expected = expected[0]

            actual = ivy.to_numpy(result)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
    finally:
        ivy.unset_backend()
        np.random.set_state(original_state)
