"""
https://github.com/data-apis/array-api/blob/master/spec/API_specification/broadcasting.md
"""

import pytest

from ..algos import BroadcastError, _broadcast_shapes


@pytest.mark.parametrize(
    "shape1, shape2, expected",
    [
        [(8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)],
        [(5, 4), (1,), (5, 4)],
        [(5, 4), (4,), (5, 4)],
        [(15, 3, 5), (15, 1, 5), (15, 3, 5)],
        [(15, 3, 5), (3, 5), (15, 3, 5)],
        [(15, 3, 5), (3, 1), (15, 3, 5)],
    ],
)
def test_broadcast_shapes(shape1, shape2, expected):
    assert _broadcast_shapes(shape1, shape2) == expected


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        [(3,), (4,)],  # dimension does not match
        [(2, 1), (8, 4, 3)],  # second dimension does not match
        [(15, 3, 5), (15, 3)],  # singleton dimensions can only be prepended
    ],
)
def test_broadcast_shapes_fails_on_bad_shapes(shape1, shape2):
    with pytest.raises(BroadcastError):
        _broadcast_shapes(shape1, shape2)
