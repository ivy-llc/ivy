# global
from hypothesis import assume, strategies as st
import math
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method, handle_test
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy.array import Array

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
        x2 = ivy.nested_map(
            x2[0], lambda x: abs(x), include_derived={list: True}, shallow=False
        )
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


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_array_property_data(
    dtype_x,
    ground_truth_backend,
):
    _, data = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    ret = helpers.flatten_and_to_np(ret=x.data)
    ret_gt = helpers.flatten_and_to_np(ret=data)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend=ground_truth_backend,
    )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_array_property_dtype(
    dtype_x,
):
    _, data = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    ivy.assertions.check_equal(x.dtype, ivy.dtype(data))


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_array_property_device(
    dtype_x,
):
    _, data = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    ivy.assertions.check_equal(x.device, ivy.dev(data))


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
    ),
)
def test_array_property_ndim(
    dtype_x,
):
    _, data, input_shape = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    ivy.assertions.check_equal(x.ndim, len(input_shape))


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
    ),
)
def test_array_property_shape(
    dtype_x,
):
    _, data, input_shape = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    ivy.assertions.check_equal(x.shape, ivy.Shape(input_shape))


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
        min_num_dims=1,
    ),
)
def test_array_property_size(
    dtype_x,
):
    _, data, input_shape = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    size_gt = 1
    for dim in input_shape:
        size_gt *= dim
    ivy.assertions.check_equal(x.size, size_gt)


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
    ),
)
def test_array_property_mT(
    dtype_x,
    ground_truth_backend,
):
    _, data = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    ret = helpers.flatten_and_to_np(ret=x.mT)
    ret_gt = helpers.flatten_and_to_np(ret=ivy.matrix_transpose(data))
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend=ground_truth_backend,
    )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=2,
    ),
)
def test_array_property_T(
    dtype_x,
    ground_truth_backend,
):
    _, data = dtype_x
    data = ivy.native_array(data[0])
    x = Array(data)
    ret = helpers.flatten_and_to_np(ret=x.T)
    ret_gt = helpers.flatten_and_to_np(ret=ivy.matrix_transpose(data))
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend=ground_truth_backend,
    )


@handle_method(method_tree="Array.__getitem__", query_dtype_and_x=_getitem_setitem())
def test_array__getitem__(
    query_dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    query, x_dtype = query_dtype_and_x
    dtype, x = x_dtype
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"query": query},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__setitem__",
    query_dtype_and_x=_getitem_setitem(),
    val=st.floats(min_value=-6, max_value=6),
)
def test_array__setitem__(
    query_dtype_and_x,
    val,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    query, x_dtype = query_dtype_and_x
    dtype, x = x_dtype
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"query": query, "val": val},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__pos__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__pos__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__neg__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__pow__",
    dtype_and_x=_pow_helper(),
)
def test_array__pow__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]):
        x[1] = np.abs(x[1])
    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"power": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rpow__",
    dtype_and_x=_pow_helper(),
)
def test_array__rpow__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]):
        x[1] = np.abs(x[1])
    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"power": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ipow__",
    dtype_and_x=_pow_helper(),
)
def test_array__ipow__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]):
        x[1] = np.abs(x[1])
    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"power": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__add__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__radd__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__iadd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__iadd__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__sub__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__rsub__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__isub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__isub__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__mul__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__rmul__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__imul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__imul__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__mod__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rmod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__rmod__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__imod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__imod__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__divmod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__divmod__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rdivmod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__rdivmod__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__truediv__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rtruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__rtruediv__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__itruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__itruediv__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__floordiv__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rfloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__rfloordiv__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ifloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__ifloordiv__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__matmul__",
    x1=_get_first_matrix_and_dtype(),
    x2=_get_second_matrix_and_dtype(),
)
def test_array__matmul__(
    x1,
    x2,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype1, x1 = x1
    dtype2, x2 = x2
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x1},
        init_input_dtypes=dtype1,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype2,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x2},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rmatmul__",
    x1=_get_first_matrix_and_dtype(),
    x2=_get_second_matrix_and_dtype(),
)
def test_array__rmatmul__(
    x1,
    x2,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype1, x1 = x1
    dtype2, x2 = x2
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x2},
        init_input_dtypes=dtype1,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype2,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x1},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__imatmul__",
    x1=_get_first_matrix_and_dtype(),
    x2=_get_second_matrix_and_dtype(),
)
def test_array__imatmul__(
    x1,
    x2,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype1, x1 = x1
    dtype2, x2 = x2
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x1},
        init_input_dtypes=dtype1,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype2,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x2},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__abs__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__abs__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=[],
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__float__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
    ),
)
def test_array__float__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=[],
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__int__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
        min_value=-1e15,
        max_value=1e15,
    ),
)
def test_array__int__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=[],
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__bool__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        max_num_dims=0,
        min_value=0,
        max_value=1,
    ),
)
def test_array__bool__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=[],
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__lt__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__le__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__eq__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ne__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__gt__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ge__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__and__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__and__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__rand__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__iand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__iand__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__or__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ror__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ror__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ior__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ior__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
    ),
)
def test_array__invert__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__xor__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rxor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__rxor__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ixor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ixor__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__lshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__lshift__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    x[1] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rlshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__rlshift__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    x[0] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__ilshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__ilshift__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    x[1] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        shared_dtype=True,
    ),
)
def test_array__rshift__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    x[1] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__rrshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__rrshift__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    x[0] = np.asarray(np.clip(x[0], 0, np.iinfo(dtype[0]).bits - 1), dtype=dtype[0])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__irshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__irshift__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    x[1] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__deepcopy__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_array__deepcopy__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__len__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=2,
        min_num_dims=1,
    ),
)
def test_array__len__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Array.__iter__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=2,
        min_num_dims=1,
    ),
)
def test_array__iter__(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args: pf.NumPositionalArg,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        method_input_dtypes=dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )
