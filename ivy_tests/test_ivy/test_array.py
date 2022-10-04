# global
from copy import deepcopy
from hypothesis import assume, given, strategies as st
import math
import numpy as np

# local
from ivy.array import Array
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


_zero = np.asarray(0, dtype="uint8")
_one = np.asarray(1, dtype="uint8")


def _not_too_close_to_zero(x):
    f = np.vectorize(lambda item: item + (_one if np.isclose(item, 0) else _zero))
    return f(x)


@st.composite
def _pow_helper(draw, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    dtype1, x1 = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            small_abs_safety_factor=4,
            large_abs_safety_factor=4,
        )
    )
    dtype1 = dtype1[0]

    def cast_filter(dtype1_x1_dtype2):
        dtype1, _, dtype2 = dtype1_x1_dtype2
        if (ivy.as_ivy_dtype(dtype1), ivy.as_ivy_dtype(dtype2)) in ivy.promotion_table:
            return True
        return False

    dtype1, x1, dtype2 = draw(
        helpers.get_castable_dtype(draw(available_dtypes), dtype1, x1).filter(
            cast_filter
        )
    )
    if ivy.is_int_dtype(dtype2):
        max_val = ivy.iinfo(dtype2).max
    else:
        max_val = ivy.finfo(dtype2).max
    max_x1 = np.max(np.abs(x1[0]))
    if max_x1 in [0, 1]:
        max_value = None
    else:
        max_value = int(math.log(max_val) / math.log(max_x1))
        if abs(max_value) > abs(max_val) / 40 or max_value < 0:
            max_value = None
    dtype2, x2 = draw(
        helpers.dtype_and_values(
            small_abs_safety_factor=12,
            large_abs_safety_factor=12,
            safety_factor_scale="log",
            max_value=max_value,
            dtype=[dtype2],
        )
    )
    dtype2 = dtype2[0]
    if "int" in dtype2:
        x2 = ivy.nested_map(x2[0], lambda x: abs(x), include_derived={list: True})
    return [dtype1, dtype2], [x1, x2]


# __matmul__ helper
@st.composite
def _get_first_matrix_and_dtype(draw, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(available_dtypes)),
            key="shared_dtype",
        )
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(helpers.ints(min_value=2, max_value=4))
    batch_shape = draw(
        st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=3), key="shape")
    )
    return [input_dtype], draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [random_size, shared_size]),
            min_value=2,
            max_value=5,
        )
    )


# __matmul__ helper
@st.composite
def _get_second_matrix_and_dtype(draw, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(available_dtypes)),
            key="shared_dtype",
        )
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(helpers.ints(min_value=2, max_value=4))
    batch_shape = draw(
        st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=3), key="shape")
    )
    return [input_dtype], draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [shared_size, random_size]),
            min_value=2,
            max_value=5,
        )
    )


# getitem and setitem helper
@st.composite
def _getitem_setitem(draw, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    arr_size = draw(helpers.ints(min_value=2, max_value=10))
    x = draw(
        helpers.dtype_and_values(available_dtypes=available_dtypes, shape=(arr_size,))
    )
    index = draw(helpers.ints(min_value=0, max_value=arr_size - 1))
    return index, x


# __getitem__
@handle_cmd_line_args
@given(query_dtype_and_x=_getitem_setitem())
def test_array__getitem__(
    query_dtype_and_x,
):
    query, x_dtype = query_dtype_and_x
    _, x = x_dtype
    data = Array(x[0])
    ret = data.__getitem__(query)
    np_ret = x[0].__getitem__(query)
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __setitem__
@handle_cmd_line_args
@given(
    query_dtype_and_x=_getitem_setitem(),
    val=st.floats(min_value=-6, max_value=6),
)
def test_array__setitem__(
    query_dtype_and_x,
    val,
):
    query, x_dtype = query_dtype_and_x
    _, x = x_dtype
    data = Array(x[0])
    ret = data.__setitem__(query, val)
    np_ret = x[0].__setitem__(query, val)
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __pos__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__pos__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = +data
    np_ret = +x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __neg__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__neg__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = -data
    np_ret = -x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __pow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
)
def test_array__pow__(
    dtype_and_x,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]):
        x[1] = np.abs(x[1])
    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    data = Array(x[0])
    power = Array(x[1])
    ret = pow(data, power)
    np_ret = pow(x[0], x[1])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rpow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
)
def test_array__rpow__(
    dtype_and_x,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]) and ivy.is_int_dtype(dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    data = Array(x[1])
    power = Array(x[0])
    ret = data.__rpow__(power)
    np_ret = x[1].__rpow__(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ipow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
)
def test_array__ipow__(
    dtype_and_x,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when data is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]) and ivy.is_int_dtype(dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    data = Array(x[0])
    power = Array(x[1])
    ret = data.__ipow__(power)
    np_ret = pow(x[0], x[1])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __add__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__add__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data + other
    np_ret = x[0] + x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __radd__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__radd__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__radd__(other)
    np_ret = x[0] + x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __iadd__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__iadd__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__iadd__(other)
    np_ret = x[0] + x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __sub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__sub__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data - other
    np_ret = x[0] - x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rsub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__rsub__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rsub__(other)
    np_ret = x[1] - x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __isub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__isub__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__isub__(other)
    np_ret = x[0] - x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __mul__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__mul__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data * other
    np_ret = x[0] * x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rmul__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__rmul__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rmul__(other)
    np_ret = x[0] * x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __imul__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__imul__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__imul__(other)
    np_ret = x[0] * x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __mod__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__mod__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data % other
    np_ret = x[0] % x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rmod__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__rmod__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rmod__(other)
    np_ret = x[1] % x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __imod__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__imod__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__imod__(other)
    np_ret = x[0] % x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __divmod__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__divmod__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = divmod(data, other)
    np_ret = divmod(x[0], x[1])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rdivmod__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__rdivmod__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rdivmod__(other)
    np_ret = divmod(x[1], x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __truediv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__truediv__(
    dtype_and_x,
):
    _, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    data = Array(x[0])
    other = Array(x[1])
    ret = data / other
    np_ret = x[0] / x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rtruediv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__rtruediv__(
    dtype_and_x,
):
    _, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rtruediv__(other)
    np_ret = x[1] / x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __itruediv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__itruediv__(
    dtype_and_x,
):
    _, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__itruediv__(other)
    np_ret = x[0] / x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __floordiv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        shared_dtype=True,
        safety_factor_scale="linear",
    ),
)
def test_array__floordiv__(
    dtype_and_x,
):
    _, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    data = Array(x[0])
    other = Array(x[1])
    ret = data // other
    np_ret = x[0] // x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rfloordiv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        shared_dtype=True,
        safety_factor_scale="linear",
    ),
)
def test_array__rfloordiv__(
    dtype_and_x,
):
    _, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rfloordiv__(other)
    np_ret = x[1] // x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ifloordiv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        shared_dtype=True,
        safety_factor_scale="linear",
    ),
)
def test_array__ifloordiv__(
    dtype_and_x,
):
    _, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__ifloordiv__(other)
    np_ret = x[0] // x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __matmul__
@handle_cmd_line_args
@given(
    x1=_get_first_matrix_and_dtype(),
    x2=_get_second_matrix_and_dtype(),
)
def test_array__matmul__(
    x1,
    x2,
):
    _, x1 = x1
    _, x2 = x2
    data = Array(x1)
    other = Array(x2)
    ret = data @ other
    np_ret = x1 @ x2
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rmatmul__
@handle_cmd_line_args
@given(
    x1=_get_second_matrix_and_dtype(),
    x2=_get_first_matrix_and_dtype(),
)
def test_array__rmatmul__(
    x1,
    x2,
):
    _, x1 = x1
    _, x2 = x2
    data = Array(x1)
    other = Array(x2)
    ret = data.__rmatmul__(other)
    np_ret = x2 @ x1
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __imatmul__
@handle_cmd_line_args
@given(
    x1=_get_first_matrix_and_dtype(),
    x2=_get_second_matrix_and_dtype(),
)
def test_array__imatmul__(
    x1,
    x2,
):
    _, x1 = x1
    _, x2 = x2
    data = Array(x1)
    other = Array(x2)
    ret = data.__imatmul__(other)
    np_ret = x1 @ x2
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __abs__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    )
)
def test_array__abs__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = abs(data)
    np_ret = abs(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __float__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
    )
)
def test_array__float__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = float(data)
    np_ret = float(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __int__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
    )
)
def test_array__int__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = int(data)
    np_ret = int(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __bool__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        max_num_dims=0,
        min_value=0,
        max_value=1,
    )
)
def test_array__bool__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = bool(data)
    np_ret = bool(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __lt__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__lt__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data < other
    np_ret = x[1] < x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __le__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__le__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data <= other
    np_ret = x[1] <= x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __eq__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__eq__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data == other
    np_ret = x[1] == x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ne__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ne__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data != other
    np_ret = x[1] != x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __gt__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__gt__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data > other
    np_ret = x[1] > x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ge__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ge__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data >= other
    np_ret = x[1] >= x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __and__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__and__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data & other
    np_ret = x[0] & x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rand__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__rand__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rand__(other)
    np_ret = x[1] & x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __iand__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__iand__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__iand__(other)
    np_ret = x[1] & x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __or__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__or__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data | other
    np_ret = x[0] | x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ror__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ror__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__ror__(other)
    np_ret = x[1] | x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ior__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ior__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__ior__(other)
    np_ret = x[0] | x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __invert__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        shared_dtype=True,
    ),
)
def test_array__invert__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = ~data
    np_ret = ~x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __xor__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__xor__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data ^ other
    np_ret = x[0] ^ x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rxor__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__rxor__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rxor__(other)
    np_ret = x[1] ^ x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ixor__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ixor__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__ixor__(other)
    np_ret = x[0] ^ x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __lshift__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        shared_dtype=True,
    ),
)
def test_array__lshift__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data << other
    np_ret = x[0] << x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rlshift__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        shared_dtype=True,
    ),
)
def test_array__rlshift__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rlshift__(other)
    np_ret = x[1] << x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ilshift__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        shared_dtype=True,
    ),
)
def test_array__ilshift__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__ilshift__(other)
    np_ret = x[0] << x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rshift__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        shared_dtype=True,
    ),
)
def test_array__rshift__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data >> other
    np_ret = x[0] >> x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rrshift__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        shared_dtype=True,
    ),
)
def test_array__rrshift__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rrshift__(other)
    np_ret = x[1] >> x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __irshift__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        shared_dtype=True,
    ),
)
def test_array__irshift__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__irshift__(other)
    np_ret = x[0] >> x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __deepcopy__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_array__deepcopy__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = data.__deepcopy__()
    py_ret = deepcopy(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=py_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __len__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=2,
        min_num_dims=1,
    ),
)
def test_array__len__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = len(data)
    py_ret = len(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=py_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __iter__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=2,
        min_num_dims=1,
    ),
)
def test_array__iter__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    ret = data.__iter__()
    np_ret = iter(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )
