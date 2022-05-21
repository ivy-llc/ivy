"""Collection of tests for unified general functions."""

# global
import pytest
import numpy as np
from numbers import Number

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers
from ivy.func_wrapper import _wrap_functions, _unwrap_functions


# Tests #
# ------#

# in-place

# functions to compile
def _fn_1(x, with_non_compiled: bool = False):
    for _ in range(100000):
        pass
    if with_non_compiled:
        (x + 3) * 4  # ops not to be compiled into the graph
    return x**2


def _fn_2(x, with_non_compiled: bool = False):
    for _ in range(100000):
        pass
    if with_non_compiled:
        (x + 3) * 4  # ops not to be compiled into the graph
    return (x + 10) ** 0.5 - 5


@pytest.mark.parametrize("x", [[1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
@pytest.mark.parametrize("with_array_caching", [True, False])
def test_compile(x, dtype, tensor_fn, with_array_caching, device, call):
    # smoke test
    if (
        (isinstance(x, Number) or len(x) == 0)
        and tensor_fn == helpers.var_fn
        and call is helpers.mx_call
    ):
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    _unwrap_functions()

    # function 1
    comp_fn = ivy.compile(_fn_1)
    # type test
    assert callable(comp_fn)
    # value test
    x = tensor_fn(x, dtype, device)
    non_compiled_return = _fn_1(x)
    x = tensor_fn(x, dtype, device)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return),
                       ivy.to_numpy(compiled_return))

    # function 2
    comp_fn = ivy.compile(_fn_2)
    # type test
    assert callable(comp_fn)
    # value test
    x = tensor_fn(x, dtype, device)
    non_compiled_return = _fn_2(x)
    x = tensor_fn(x, dtype, device)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return),
                       ivy.to_numpy(compiled_return))

    _wrap_functions()
