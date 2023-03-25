import pytest
from hypothesis import given

from .. import dtype_helpers as dh
from .. import hypothesis_helpers as hh
from .. import _array_module as xp
from .._array_module import _UndefinedStub


# e.g. PyTorch only supports uint8 currently
@pytest.mark.skipif(isinstance(xp.uint8, _UndefinedStub), reason="uint8 not defined")
@pytest.mark.skipif(
    not all(isinstance(d, _UndefinedStub) for d in dh.uint_dtypes[1:]),
    reason="uints defined",
)
@given(hh.mutually_promotable_dtypes(dtypes=dh.uint_dtypes))
def test_mutually_promotable_dtypes(pair):
    assert pair == (xp.uint8, xp.uint8)
