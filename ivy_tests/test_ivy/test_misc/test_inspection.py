# global
import pytest
from typing import List, Tuple, Dict, Optional, Union

# local
import ivy


def _fn0(xs: Optional[List[ivy.Array]] = None):
    return xs


def _fn1(
    a: Union[ivy.Array, ivy.NativeArray],
    b: str = "hello",
    c: int = None,
    d: ivy.NativeArray = None,
):
    return a, b, c, d


def _fn2(
    a: Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
    bs: Tuple[str] = ("a", "b", "c"),
    cs: Dict[str, ivy.Array] = None,
):
    return a, bs, cs


@pytest.mark.parametrize(
    "fn_n_spec",
    [
        (_fn0, [[(0, "xs"), "optional", int]]),
        (_fn1, [[(0, "a")], [(3, "d"), "optional"]]),
        (_fn2, [[(0, "a"), int], [(2, "cs"), "optional", str]]),
    ],
)
def test_fn_array_spec(fn_n_spec, backend_fw):
    ivy.set_backend(backend_fw)
    fn, spec = fn_n_spec
    assert ivy.fn_array_spec(fn) == spec
    ivy.previous_backend()
