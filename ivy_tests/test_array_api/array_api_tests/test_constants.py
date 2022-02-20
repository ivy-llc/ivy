import math
from typing import Any, SupportsFloat

import pytest

from . import dtype_helpers as dh
from ._array_module import mod as xp
from .typing import Array

pytestmark = pytest.mark.ci


def assert_scalar_float(name: str, c: Any):
    assert isinstance(c, SupportsFloat), f"{name}={c!r} does not look like a float"


def assert_0d_float(name: str, x: Array):
    assert dh.is_float_dtype(
        x.dtype
    ), f"xp.asarray(xp.{name})={x!r}, but should have float dtype"


@pytest.mark.parametrize("name, n", [("e", math.e), ("pi", math.pi)])
def test_irrational_numbers(name, n):
    assert hasattr(xp, name)
    c = getattr(xp, name)
    assert_scalar_float(name, c)
    floor = math.floor(n)
    assert c > floor, f"xp.{name}={c!r} <= {floor}"
    ceil = math.ceil(n)
    assert c < ceil, f"xp.{name}={c!r} >= {ceil}"
    x = xp.asarray(c)
    assert_0d_float("name", x)


def test_inf():
    assert hasattr(xp, "inf")
    assert_scalar_float("inf", xp.inf)
    assert math.isinf(xp.inf)
    assert xp.inf > 0, "xp.inf not greater than 0"
    x = xp.asarray(xp.inf)
    assert_0d_float("inf", x)
    assert xp.isinf(x), "xp.isinf(xp.asarray(xp.inf))=False"


def test_nan():
    assert hasattr(xp, "nan")
    assert_scalar_float("nan", xp.nan)
    assert math.isnan(xp.nan)
    assert xp.nan != xp.nan, "xp.nan should not have equality with itself"
    x = xp.asarray(xp.nan)
    assert_0d_float("nan", x)
    assert xp.isnan(x), "xp.isnan(xp.asarray(xp.nan))=False"
