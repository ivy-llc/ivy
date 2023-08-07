import pytest

from ivy_tests.test_ivy.helpers.decorators.backend.function_decorator_backend import (
    BackendFunctionHandler,
)
from hypothesis import strategies as st


@pytest.mark.backend_independent
def test_returns_no_hypothesis_func():
    def fake_test_fn():
        pass

    original_fn_id = id(fake_test_fn)
    wrapped_fn_id = id(
        BackendFunctionHandler(
            fn_tree="functional.ivy.add",
            ground_truth_backend="tensorflow",
        )(fake_test_fn)
    )

    assert original_fn_id == wrapped_fn_id


@pytest.mark.backend_independent
def test_returns_hypothesis_func():
    def fake_test_fn():
        pass

    wrapped_fn = BackendFunctionHandler(
        fn_tree="functional.ivy.add",
        ground_truth_backend="tensorflow",
        x=st.integers(),
    )(fake_test_fn)

    assert id(fake_test_fn) != id(wrapped_fn)
    assert hasattr(wrapped_fn, "hypothesis")
    assert wrapped_fn.is_hypothesis_test


@pytest.mark.backend_independent
def test_has_test_attrs():
    def fake_test_fn():
        pass

    wrapped_fn = BackendFunctionHandler(
        fn_tree="functional.ivy.add",
        ground_truth_backend="tensorflow",
        x=st.integers(),
    )(fake_test_fn)

    assert hasattr(wrapped_fn, "_is_ivy_backend_test")
