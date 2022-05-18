"""Collection of tests for unified dtype functions."""

# global
import pytest
from numbers import Number
import ivy_tests.test_ivy.helpers as helpers

# local
import ivy
import ivy.functional.backends.numpy
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy.functional.backends.mxnet


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


# dtype bits
@pytest.mark.parametrize("x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_dtype_bits(x, dtype, tensor_fn, device, call):
    # smoke test
    if ivy.invalid_dtype(dtype):
        pytest.skip()
    if (
        (isinstance(x, Number) or len(x) == 0)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, device)
    ret = ivy.dtype_bits(ivy.dtype(x))
    # type test
    assert isinstance(ret, int)
    assert ret in [1, 8, 16, 32, 64]


# dtype_to_str
@pytest.mark.parametrize("x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "float64", "int8", "int16", "int32", "int64", "bool"],
)
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_dtype_to_str(x, dtype, tensor_fn, device, call):
    # smoke test
    if call is helpers.mx_call and dtype == "int16":
        # mxnet does not support int16
        pytest.skip()
    if call is helpers.jnp_call and dtype in ["int64", "float64"]:
        # jax does not support int64 or float64 arrays
        pytest.skip()
    if (
        (isinstance(x, Number) or len(x) == 0)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, device)
    dtype_as_str = ivy.dtype(x, as_str=True)
    dtype_to_str = ivy.dtype_to_str(ivy.dtype(x))
    # type test
    assert isinstance(dtype_as_str, str)
    assert isinstance(dtype_to_str, str)
    # value test
    assert dtype_to_str == dtype_as_str


# dtype_from_str
@pytest.mark.parametrize("x", [1, [], [1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "float64", "int8", "int16", "int32", "int64", "bool"],
)
@pytest.mark.parametrize("tensor_fn", [ivy.array])
def test_dtype_from_str(x, dtype, tensor_fn, device, call):
    # smoke test
    if call is helpers.mx_call and dtype == "int16":
        # mxnet does not support int16
        pytest.skip()
    if call is helpers.jnp_call and dtype in ["int64", "float64"]:
        # jax does not support int64 or float64 arrays
        pytest.skip()
    if (
        (isinstance(x, Number) or len(x) == 0)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x, dtype, device)
    dt0 = ivy.dtype_from_str(ivy.dtype(x, as_str=True))
    dt1 = ivy.dtype(x)
    # value test
    assert dt0 is dt1

# Still to Add #
# ---------------#

# astype
# broadcast_arrays
# broadcast_to
# can_cast
# finfo
# iinfo
# result_type
