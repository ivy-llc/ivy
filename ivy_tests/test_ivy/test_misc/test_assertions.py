import sys
import os
import contextlib
import pytest
from ivy.utils.assertions import check_greater, check_less


@pytest.mark.parametrize(
    "x1, x2, allow_equal",
    [
        (5, 10, False),
        (10, 5, False),
        (5, 5, True),
        (10, 5, True),
    ],
)
def test_check_less(x1, x2, allow_equal):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_less(x1, x2, allow_equal)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if x1 > x2 and allow_equal:
        assert "lesser than or equal" in lines.strip()

    if x1 > x2 and not allow_equal:
        assert "lesser than" in lines.strip()

    if x1 < x2:
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "x1, x2, allow_equal",
    [
        (5, 10, False),
        (10, 5, False),
        (5, 5, True),
        (10, 5, True),
    ],
)
def test_check_greater(x1, x2, allow_equal):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_greater(x1, x2, allow_equal)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if x1 < x2 and allow_equal:
        assert "greater than or equal" in lines.strip()

    if x1 < x2 and not allow_equal:
        assert "greater than" in lines.strip()

    if x1 > x2:
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)
