import ivy
import pytest
from ivy.func_wrapper import handle_array_like
from typing import Union, Tuple, List, Sequence


@handle_array_like
def fn1(x: Union[ivy.Array, Tuple[int, int]]):
    return x


@handle_array_like
def fn2(x: Union[ivy.Array, ivy.NativeArray]):
    return x


@handle_array_like
def fn3(x: List[ivy.Array]):
    return x


@handle_array_like
def fn4(x: Union[Sequence[ivy.Array], ivy.Array]):
    return x


@pytest.mark.parametrize(("fn", "x", "expected_type"),
                         [(fn1, (1, 2), tuple),
                          (fn2, (1, 2), ivy.Array),
                          (fn2, [1, 2], ivy.Array),
                          (fn3, [1, 2], list),
                          (fn4, [1, 2], list)])
def test_handle_array_like(fn, x, expected_type):
    assert isinstance(fn(x), expected_type)
