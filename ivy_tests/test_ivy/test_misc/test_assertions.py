import sys
import os
import contextlib
import pytest
from ivy.utils.assertions import (
    check_all,
    check_all_or_any_fn,
    check_any,
    check_dimensions,
    check_elem_in_list,
    check_equal,
    check_exists,
    check_false,
    check_fill_value_and_dtype_are_compatible,
    check_gather_nd_input_valid,
    check_greater,
    check_inplace_sizes_valid,
    check_isinstance,
    check_kernel_padding_size,
    check_less,
    check_same_dtype,
    check_shapes_broadcastable,
    check_true,
    check_unsorted_segment_min_valid_params,
)
from ivy.utils.assertions import _check_jax_x64_flag
import ivy


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


@pytest.mark.parametrize(
    "x1, x2, inverse",
    [
        (5, 10, False),
        (10, 10, False),
        (5, 5, True),
        (10, 5, True),
    ],
)
def test_check_equal(x1, x2, inverse):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_equal(x1, x2, inverse)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if inverse:
        if x1 == x2:
            assert "must not be equal" in lines.strip()

        if x1 != x2:
            assert not lines.strip()

    if not inverse:
        if x1 != x2:
            assert "must be equal" in lines.strip()

        if x1 == x2:
            assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "x, allowed_types",
    [(5.0, float), (ivy.array(5), type(ivy.array(8))), (5, float), ([5, 10], tuple)],
)
def test_check_isinstance(x, allowed_types):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_isinstance(x, allowed_types)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if not isinstance(x, allowed_types):
        assert "must be one of the" in lines.strip()

    if isinstance(x, allowed_types):
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "x, inverse",
    [(None, False), ([], False), (None, True), ("abc", True)],
)
def test_check_exists(x, inverse):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_exists(x, inverse)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if not inverse:
        if x is None:
            assert "must not be" in lines.strip()

        if x:
            assert not lines.strip()

    if inverse:
        if x is None:
            assert not lines.strip()

        if x:
            assert "must be None" in lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "elem, list, inverse",
    [
        (1, [1, 2], False),
        ("a", [1, 2], False),
        (1, [2, 3], True),
        (0, ["a", "b", "c"], True),
    ],
)
def test_check_elem_in_list(elem, list, inverse):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_elem_in_list(elem, list, inverse)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if not inverse:
        if elem not in list:
            assert "must be one" in lines.strip()

        if elem in list:
            assert not lines.strip()

    if inverse:
        if elem not in list:
            assert not lines.strip()

        if elem in list:
            assert "must not be one" in lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "expression",
    [
        (True),
        "a",
        (None),
        (False),
    ],
)
def test_check_true(expression):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_true(expression)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if not expression:
        assert "True" in lines.strip()

    if expression:
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "expression",
    [
        (True),
        "a",
        (None),
        (False),
    ],
)
def test_check_false(expression):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_false(expression)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if not expression:
        assert not lines.strip()

    if expression:
        assert "False" in lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "results",
    [
        ([0, 1, 2]),
        ([True, False]),
        ([True, True]),
    ],
)
def test_check_all(results):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_all(results)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if not all(results):
        assert "one" in lines.strip()

    if all(results):
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "results",
    [([0, 1, 2]), ([False, False]), ([True, False]), ([0, False])],
)
def test_check_any(results):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_any(results)
    except Exception as e:
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if not any(results):
        assert "all" in lines.strip()

    if all(results):
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "args, fn, type, limit",
    [
        # INVALID CASES
        ((1, 2, 0), ivy.array, "all", [3]),
        ((0, 0), ivy.array, "all", [2]),
        ((1, 1), ivy.array, "any", [3]),
        ((0, 0, 1), ivy.array, "any", [3]),
        ((1, 0, 1), ivy.array, "all_any", [3]),
        # VALID
        ((1, 1), ivy.array, "any", [2]),
        ((0, 1), ivy.array, "any", [1]),
        ((1, 1, 2), ivy.array, "all", [3]),
    ],
)
def test_check_all_or_any_fn(args, fn, type, limit):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_all_or_any_fn(*args, fn=fn, type=type, limit=limit)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)
    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if type == "all" or type == "any":
        if "e" in local_vars.keys():
            assert "args must exist according to" in lines.strip()
        else:
            assert not lines.strip()

    else:
        assert "type must be all or any" in lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "x1, x2",
    [
        (ivy.array([1, 2, 3]), ivy.array([4, 5, 6])),
        (ivy.array([1.0, 2.0, 3.0]), ivy.array([4, 5, 6])),
        (ivy.array([1, 2, 3]), ivy.array([4j, 5 + 1j, 6])),
        (ivy.array([1j]), ivy.array([2, 3 + 4j])),
    ],
)
def test_check_same_dtype(x1, x2):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_same_dtype(x1, x2)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert "same dtype" in lines.strip()

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "fill_value, dtype",
    [
        # INVALID CASES
        (1.0, ivy.int16),
        (1, ivy.float16),
        (1, ivy.complex64),
        # VALID
        (1j, ivy.complex64),
        (1.0, ivy.complex64),
        (1.0, ivy.float16),
        (1, ivy.int16),
    ],
)
def test_check_fill_value_and_dtype_are_compatible(fill_value, dtype):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_fill_value_and_dtype_are_compatible(fill_value, dtype)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert "not compatible" in lines.strip()

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "data, segment_ids, num_segments",
    [
        # INVALID CASES
        (ivy.array([1, 2, 3]), ivy.array([0, 1, 0], dtype=ivy.int32), 2.0),
        (ivy.array([1, 2, 3]), ivy.array([0, 1, 0], dtype=ivy.int32), 0),
        (ivy.array([1, 2, 3]), ivy.array([0, 1, 0], dtype=ivy.int32), -2),
        (ivy.array([1, 2, 3]), ivy.array([0, 1, 0], dtype=ivy.int32), 0),
        (ivy.array([1, 2, 3]), ivy.array([0.0, 1.0, 0.0], dtype=ivy.float16), 0),
        (ivy.array([1, 2]), ivy.array([0, 1, 0], dtype=ivy.int32), 0),
        (ivy.array([1, 2, 3]), ivy.array([0, 1], dtype=ivy.int32), 0),
        (ivy.array([1, 2, 3]), ivy.array([0, 1, 2], dtype=ivy.int32), 2),
        # VALID
        (
            ivy.array([1, 2, 3]),
            ivy.array([0, 1, 0], dtype=ivy.int32),
            2,
        ),
        (ivy.array([1, 2, 3]), ivy.array([0, 1, 0], dtype=ivy.int32), ivy.array([2])),
    ],
)
def test_check_unsorted_segment_min_valid_params(data, segment_ids, num_segments):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_unsorted_segment_min_valid_params(data, segment_ids, num_segments)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert (
            "num_segments must be of integer type" in lines.strip()
            or "segment_ids must have an integer dtype" in lines.strip()
            or "segment_ids should be equal to data.shape[0]" in lines.strip()
            or "is out of range" in lines.strip()
            or "num_segments must be positive" in lines.strip()
        )

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "params, indices, batch_dims",
    [
        # INVALID CASES
        (ivy.array([1, 2, 3]), ivy.array([1]), 2),
        (ivy.array([[1, 2, 3], [4, 5, 6]]), ivy.array([0, 2]), 1),
        (ivy.array([[1, 2, 3], [4, 5, 6]]), ivy.array([[0, 1], [1, 2], [2, 3]]), 1),
        (ivy.array([1, 2, 3]), ivy.array([[1, 2]]), 0),
        # VALID
        (ivy.array([1, 2, 3]), ivy.array([1]), 0),
        (ivy.array([[1, 2, 3], [4, 5, 6]]), ivy.array([0, 2]), 0),
        (ivy.array([[1, 2, 3], [4, 5, 6]]), ivy.array([[0, 1], [1, 2]]), 1),
    ],
)
def test_check_gather_nd_input_valid(params, indices, batch_dims):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_gather_nd_input_valid(params, indices, batch_dims)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert (
            "less than rank(`params`)" in lines.strip()
            or "less than rank(`indices`)" in lines.strip()
            or "dimensions must match in `params` and `indices`" in lines.strip()
            or "index innermost dimension length must be <=" in lines.strip()
        )

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "var, data",
    [
        # INVALID CASES
        (ivy.array([1]), ivy.array([1, 2])),
        (ivy.array([[1], [1], [2]]), ivy.array([1, 2])),
        # VALID
        (ivy.array([1, 2]), ivy.array([1])),
        (ivy.array([[[1]]]), ivy.array([1, 2])),
    ],
)
def test_check_inplace_sizes_valid(var, data):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_inplace_sizes_valid(var, data)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert "Could not output values of shape" in lines.strip()

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "var, data",
    [
        # INVALID CASES
        ((2, 1), (1, 2, 1)),
        ((2, 1), (3, 1)),
        # VALID
        ((1, 2), (1, 2)),
        ((1, 2), (1, 1, 1)),
    ],
)
def test_check_shapes_broadcastable(var, data):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_shapes_broadcastable(var, data)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert "Could not broadcast shape" in lines.strip()

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "x",
    [
        # INVALID CASES
        (ivy.array([1])),
        (ivy.array([])),
        # VALID
        (ivy.array([1, 2])),
        (ivy.array([[1, 2], [2, 3]])),
    ],
)
def test_check_dimensions(x):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_dimensions(x)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert "greater than one dimension" in lines.strip()

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "kernel_size, padding_size",
    [
        # INVALID CASES
        (((2, 2), ((2, 2), (1, 1)))),
        (((3, 3), ((2, 2), (1, 1)))),
        # VALID
        (((5, 5), ((1, 1), (2, 2)))),
        (((3, 3), ((1, 1), (0, 0)))),
    ],
)
def test_check_kernel_padding_size(kernel_size, padding_size):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        check_kernel_padding_size(kernel_size, padding_size)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert "less than or equal to half" in lines.strip()

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


@pytest.mark.parametrize(
    "dtype",
    [
        # INVALID CASES
        "float64",
        "int64",
        "uint64",
        "complex128"
        # VALID
        "float16",
        "float32int32",
        "int16",
        "complex64",
    ],
)
def test_check_jax_x64_flag(dtype):
    filename = "except_out.txt"
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    lines = ""
    try:
        _check_jax_x64_flag(dtype)
        local_vars = {**locals()}
    except Exception as e:
        local_vars = {**locals()}
        print(e)

    sys.stdout = orig_stdout
    f.close()

    with open(filename) as f:
        lines += f.read()

    if "e" in local_vars.keys():
        assert "output not supported while jax_enable_x64" in lines.strip()

    if "e" not in local_vars.keys():
        assert not lines.strip()

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)
