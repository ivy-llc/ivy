"""Regression tests for scalar indexing with an inferred NumPy backend."""

import numpy as np
import pytest

import ivy


@pytest.fixture
def no_backend(backend_fw):
    if backend_fw != "numpy":
        pytest.skip("NumPy backend inference regression")
    saved_backends = tuple(ivy.backend_stack)
    while ivy.backend_stack:
        ivy.unset_backend()
    try:
        yield
    finally:
        while ivy.backend_stack:
            ivy.unset_backend()
        for backend in saved_backends:
            ivy.set_backend(backend)


@pytest.mark.parametrize("dtype", ["int32", "float64"])
@pytest.mark.parametrize("api", ["index", "ivy", "native"])
@pytest.mark.parametrize(
    "query",
    [
        None,
        (),
        Ellipsis,
        (None,),
        (None, Ellipsis),
        (Ellipsis, None),
        (None, None),
        (None, Ellipsis, None),
        (None, None, Ellipsis, None),
    ],
)
def test_scalar_indexing_without_backend(no_backend, dtype, api, query):
    x = ivy.array(np.array(5, dtype=dtype))
    native = ivy.to_native(x)
    expected = native[query]

    if api == "index":
        result = x[query]
    else:
        result = ivy.get_item(native if api == "native" else x, query)

    assert ivy.current_backend_str() == ""
    assert ivy.is_ivy_array(result)
    actual = ivy.to_native(result)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.shares_memory(native, actual) == np.shares_memory(native, expected)


@pytest.mark.parametrize("copy", [None, False, True])
@pytest.mark.parametrize("query", [None, (), Ellipsis, (None, Ellipsis, None)])
def test_scalar_get_item_copy(no_backend, copy, query):
    x = ivy.array(np.array(5, dtype="int32"))
    native = ivy.to_native(x)
    expected = (native.copy() if copy else native)[query]
    result = ivy.get_item(x, query, copy=copy)
    actual = ivy.to_native(result)

    np.testing.assert_array_equal(actual, expected)
    shares_memory = np.shares_memory(native, expected)
    assert np.shares_memory(native, actual) == shares_memory
    ivy.inplace_update(x, ivy.array(7, dtype="int32"))
    np.testing.assert_array_equal(
        actual, np.full(expected.shape, 7 if shares_memory else 5)
    )
    assert ivy.current_backend_str() == ""


@pytest.mark.parametrize("api", ["index", "function"])
@pytest.mark.parametrize(
    "query",
    [
        0,
        -1,
        slice(None),
        (None, 0),
        (None, slice(None)),
        (Ellipsis, 0),
        (Ellipsis, Ellipsis),
    ],
)
def test_invalid_scalar_indexing_without_backend(no_backend, api, query):
    native = np.array(5)
    x = ivy.array(native)
    with pytest.raises(IndexError):
        native[query]
    with pytest.raises(IndexError):
        if api == "index":
            x[query]
        else:
            ivy.get_item(x, query)
    assert ivy.current_backend_str() == ""
