import ivy
import pytest
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
def test_handle_array_like_without_promotion(fn, x, expected_type):
    assert isinstance(handle_array_like_without_promotion(fn)(x), expected_type)


def test_outputs_to_ivy_arrays():
    assert isinstance(
        ivy.outputs_to_ivy_arrays(_fn1)(ivy.to_native(ivy.array([2.0]))), ivy.Array
    )
    assert ivy.outputs_to_ivy_arrays(_fn1)(ivy.array(1)) == ivy.array(1)


def _fn5(x):
    # Test input was converted to native array
    assert isinstance(x, ivy.NativeArray)


def test_inputs_to_native_arrays():
    ivy.inputs_to_native_arrays(_fn5)(ivy.array(1))


def _fn6(x):
    # Assert input was converted to Ivy Array
    assert isinstance(x, ivy.Array)


def test_inputs_to_ivy_arrays():
    ivy.inputs_to_ivy_arrays(_fn6)(ivy.native_array(1))


def _fn7(x):
    # Assert input was converted to native array
    assert isinstance(x, ivy.NativeArray)
    return x


def test_to_native_arrays_and_back():
    x = ivy.array(1.0)
    res = ivy.func_wrapper.to_native_arrays_and_back(_fn7)(x)
    assert isinstance(res, ivy.Array)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (1, 1.0),
        ([1, 3], [1.0, 3.0]),
        (
            [[[1, 0], [-2, 1.0]]],
            [[[1.0, 0.0], [-2.0, 1.0]]],
        ),
    ],
)
def test_integer_arrays_to_float(x, expected):
    # Todo: Fix dtype issue (update: fixed)
    x = ivy.array(x)
    expected = ivy.array(expected)

    assert ivy.array_equal(ivy.func_wrapper.integer_arrays_to_float(_fn1)(x), expected)
