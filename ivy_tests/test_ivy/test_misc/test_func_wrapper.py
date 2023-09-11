import numpy as np

import ivy
import pytest
from unittest.mock import patch
from ivy.func_wrapper import handle_array_like_without_promotion
from typing import Union, Tuple, List, Sequence


def _fn1(x: Union[ivy.Array, Tuple[int, int]]):
    return x


def _fn2(x: Union[ivy.Array, ivy.NativeArray]):
    return x


def _fn3(x: List[ivy.Array]):
    return x


def _fn4(x: Union[Sequence[ivy.Array], ivy.Array]):
    return x


@pytest.mark.parametrize(
    ("fn", "x", "expected_type"),
    [
        (_fn1, (1, 2), tuple),
        (_fn2, (1, 2), ivy.Array),
        (_fn2, [1, 2], ivy.Array),
        (_fn3, [1, 2], list),
        (_fn4, [1, 2], list),
    ],
)
def test_handle_array_like_without_promotion(fn, x, expected_type, backend_fw):
    ivy.set_backend(backend_fw)
    assert isinstance(handle_array_like_without_promotion(fn)(x), expected_type)
    ivy.previous_backend()


def test_outputs_to_ivy_arrays(backend_fw):
    ivy.set_backend(backend_fw)
    assert isinstance(
        ivy.outputs_to_ivy_arrays(_fn1)(ivy.to_native(ivy.array([2.0]))), ivy.Array
    )
    assert ivy.outputs_to_ivy_arrays(_fn1)(ivy.array(1)) == ivy.array(1)
    ivy.previous_backend()


def _fn5(x):
    # Test input was converted to native array
    assert isinstance(x, ivy.NativeArray)


def test_inputs_to_native_arrays(backend_fw):
    ivy.set_backend(backend_fw)
    ivy.inputs_to_native_arrays(_fn5)(ivy.array(1))
    ivy.previous_backend()


def _fn6(x):
    # Assert input was converted to Ivy Array
    assert isinstance(x, ivy.Array)


def test_inputs_to_ivy_arrays(backend_fw):
    ivy.set_backend(backend_fw)
    ivy.inputs_to_ivy_arrays(_fn6)(ivy.native_array(1))
    ivy.previous_backend()


def _fn7(x):
    # Assert input was converted to native array
    assert isinstance(x, ivy.NativeArray)
    return x


def test_to_native_arrays_and_back(backend_fw):
    ivy.set_backend(backend_fw)
    x = ivy.array(1.0)
    res = ivy.func_wrapper.to_native_arrays_and_back(_fn7)(x)
    assert isinstance(res, ivy.Array)
    ivy.previous_backend()


@pytest.mark.parametrize(
    ("x", "weight", "expected"),
    [
        ([[1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]], True),
        (
            [[1, 1], [1, 1]],
            [
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [1, 1]],
            ],
            False,
        ),
    ],
)
def test_handle_partial_mixed_function(x, weight, expected, backend_fw):
    ivy.set_backend(backend_fw)
    test_fn = "torch.nn.functional.linear"
    if ivy.current_backend_str() != "torch":
        # ivy.matmul is used inside the compositional implementation
        test_fn = "ivy.matmul"
        expected = True
    with patch(test_fn) as test_mock_function:
        ivy.linear(ivy.array(x), ivy.array(weight))
        assert test_mock_function.called == expected
    ivy.previous_backend()


@pytest.mark.parametrize(
    "array_to_update",
    [0, 1, 2, 3, 4],
)
def test_views(array_to_update, backend_fw):
    ivy.set_backend(backend_fw)
    a = ivy.random.random_normal(shape=(6,))
    a_copy = ivy.copy_array(a)
    b = a.reshape((2, 3))
    b_copy = ivy.copy_array(b)
    c = ivy.flip(b)
    c_copy = ivy.copy_array(c)
    d = ivy.rot90(c, k=3)
    d_copy = ivy.copy_array(d)
    e = ivy.split(d)
    e_copy = ivy.copy_array(e[0])
    array = (a, b, c, d, e)[array_to_update]
    if array_to_update == 4:
        for arr in array:
            arr += 1
    else:
        array += 1
    assert np.allclose(a, a_copy + 1)
    assert np.allclose(b, b_copy + 1)
    assert np.allclose(c, c_copy + 1)
    assert np.allclose(d, d_copy + 1)
    assert np.allclose(e[0], e_copy + 1)
    ivy.previous_backend()
