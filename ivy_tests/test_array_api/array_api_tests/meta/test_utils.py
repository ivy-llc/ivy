import pytest

from .. import shape_helpers as sh
from ..test_creation_functions import frange
from ..test_manipulation_functions import roll_ndindex
from ..test_signatures import extension_module


def test_extension_module_is_extension():
    assert extension_module("linalg")


def test_extension_func_is_not_extension():
    assert not extension_module("linalg.cross")


@pytest.mark.parametrize(
    "r, size, elements",
    [
        (frange(0, 1, 1), 1, [0]),
        (frange(1, 0, -1), 1, [1]),
        (frange(0, 1, -1), 0, []),
        (frange(0, 1, 2), 1, [0]),
    ],
)
def test_frange(r, size, elements):
    assert len(r) == size
    assert list(r) == elements


@pytest.mark.parametrize(
    "shape, expected",
    [((), [()])],
)
def test_ndindex(shape, expected):
    assert list(sh.ndindex(shape)) == expected


@pytest.mark.parametrize(
    "shape, axis, expected",
    [
        ((1,), 0, [(slice(None, None),)]),
        ((1, 2), 0, [(slice(None, None), slice(None, None))]),
        (
            (2, 4),
            1,
            [(0, slice(None, None)), (1, slice(None, None))],
        ),
    ],
)
def test_axis_ndindex(shape, axis, expected):
    assert list(sh.axis_ndindex(shape, axis)) == expected


@pytest.mark.parametrize(
    "shape, axes, expected",
    [
        ((), (), [[()]]),
        ((1,), (0,), [[(0,)]]),
        (
            (2, 2),
            (0,),
            [
                [(0, 0), (1, 0)],
                [(0, 1), (1, 1)],
            ],
        ),
    ],
)
def test_axes_ndindex(shape, axes, expected):
    assert list(sh.axes_ndindex(shape, axes)) == expected


@pytest.mark.parametrize(
    "shape, shifts, axes, expected",
    [
        ((1, 1), (0,), (0,), [(0, 0)]),
        ((2, 1), (1, 1), (0, 1), [(1, 0), (0, 0)]),
        ((2, 2), (1, 1), (0, 1), [(1, 1), (1, 0), (0, 1), (0, 0)]),
        ((2, 2), (-1, 1), (0, 1), [(1, 1), (1, 0), (0, 1), (0, 0)]),
    ],
)
def test_roll_ndindex(shape, shifts, axes, expected):
    assert list(roll_ndindex(shape, shifts, axes)) == expected
