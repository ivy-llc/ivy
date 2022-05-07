"""Collection of tests for unified dtype functions."""

# global
import pytest

# local
import ivy
import ivy.functional.backends.numpy
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy.functional.backends.mxnet
import ivy_tests.test_ivy.helpers as helpers


# dtype objects
def test_dtype_instances(device, call):
    assert ivy.exists(ivy.int8)
    assert ivy.exists(ivy.int16)
    assert ivy.exists(ivy.int32)
    assert ivy.exists(ivy.int64)
    assert ivy.exists(ivy.uint8)
    if ivy.current_framework_str() != "torch":
        assert ivy.exists(ivy.uint16)
        assert ivy.exists(ivy.uint32)
        assert ivy.exists(ivy.uint64)
    assert ivy.exists(ivy.float32)
    assert ivy.exists(ivy.float64)
    assert ivy.exists(ivy.bool)


# is_int_dtype
@pytest.mark.parametrize(
    "in_n_asarray_n_res",
    [
        ([1, 2], True, True),
        ([1.3, 4.2], True, False),  # array
        (2, False, True),
        (2.6, False, False),  # number
        ([[1, 2], [3, 4]], False, True),
        ([[1.1, 2.7], [3.3, 4.5]], False, False),  # list
        ([1, 2, 3, 4], False, True),
        ([1.1, 2.7, 3.3, 4.5], False, False),  # tuple
        ({"a": [1, 2], "b": [3, 4]}, False, True),  # dict
        ({"a": [1.1, 2.7], "b": [3.3, 4.5]}, False, False),
        ("int32", False, True),
        ("float32", False, False),  # dtype str
    ],
)
def test_is_int_dtype(device, call, in_n_asarray_n_res):
    x, asarray, res = in_n_asarray_n_res
    if asarray:
        x = ivy.array(x)
    assert ivy.is_int_dtype(x) is res


# is_float_dtype
@pytest.mark.parametrize(
    "in_n_asarray_n_res",
    [
        ([1, 2], True, False),
        ([1.3, 4.2], True, True),  # array
        (2, False, False),
        (2.6, False, True),  # number
        ([[1, 2], [3, 4]], False, False),
        ([[1.1, 2.7], [3.3, 4.5]], False, True),  # list
        ([1, 2, 3, 4], False, False),
        ([1.1, 2.7, 3.3, 4.5], False, True),  # tuple
        ({"a": [1, 2], "b": [3, 4]}, False, False),  # dict
        ({"a": [1.1, 2.7], "b": [3.3, 4.5]}, False, True),
        ("int32", False, False),
        ("float32", False, True),  # dtype str
    ],
)
def test_is_float_dtype(device, call, in_n_asarray_n_res):
    x, asarray, res = in_n_asarray_n_res
    if asarray:
        x = ivy.array(x)
    assert ivy.is_float_dtype(x) is res
