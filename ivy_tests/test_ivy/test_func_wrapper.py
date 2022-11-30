import ivy
import pytest
from ivy.func_wrapper import handle_array_like
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
def test_handle_array_like(fn, x, expected_type):
    assert isinstance(handle_array_like(fn)(x), expected_type)


def test_output_to_ivy_arrays():
    assert isinstance(
        ivy.outputs_to_ivy_arrays(_fn1)(ivy.to_native(ivy.array([2.]))),
        ivy.Array)
    assert ivy.outputs_to_ivy_arrays(_fn1)(ivy.array(1)) == ivy.array(1)


def _fn5(x: ivy.NativeArray):
    assert isinstance(x, ivy.NativeArray)


def test_inputs_to_native_arrays():
    ivy.inputs_to_native_arrays(_fn5)(ivy.array(1))


def _fn6(x: ivy.Array):
    assert isinstance(x, ivy.Array)


def test_inputs_to_ivy_arrays():
    ivy.inputs_to_ivy_arrays(_fn6)(ivy.native_array(1))
