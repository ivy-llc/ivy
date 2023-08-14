# global
from hypothesis import assume, strategies as st
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method, handle_test, update_backend
from ivy_tests.test_ivy.test_functional.test_core.test_elementwise import (
    not_too_close_to_zero,
    pow_helper,
)
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
)

CLASS_TREE = "ivy.array"


def test_array_function():
    HANDLED_FUNCTIONS = {}

    class MyArray:
        def __init__(self, data=None):
            self.data = data

        def __ivy_array_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all(
                issubclass(t, (MyArray, ivy.Array, ivy.NativeArray)) for t in types
            ):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(ivy_function):
        """Register an __ivy_array_function__ implementation for MyArray objects."""

        def decorator(func):
            HANDLED_FUNCTIONS[ivy_function] = func
            return func

        return decorator

    @implements(ivy.abs)
    def _(my_array, ivy_array):
        my_array.data = abs(my_array.data)
        ivy_array = ivy.abs(ivy_array)
        return (my_array, ivy_array)

    x = MyArray(-3)
    y = ivy.array([1, -1])
    xy = _(x, ivy_array=y)
    x1 = xy[0]
    y1 = xy[1]
    assert x1.data == 3
    assert all(y1 == ivy.array([1, 1]))


# TODO: avoid using dummy fn_tree in property tests


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_array_property_data(
    dtype_x,
    backend_fw,
    test_flags,
):
    _, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ret = helpers.flatten_and_to_np(ret=x.data, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(ret=data, backend=backend_fw)
        helpers.value_test(
            ret_np_flat=ret,
            ret_np_from_gt_flat=ret_gt,
            backend=backend_fw,
            ground_truth_backend=test_flags.ground_truth_backend,
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_array_property_dtype(
    dtype_x,
    backend_fw,
):
    _, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ivy_backend.utils.assertions.check_equal(
            x.dtype, ivy_backend.dtype(data), as_array=False
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_array_property_device(
    dtype_x,
    backend_fw,
):
    _, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ivy_backend.utils.assertions.check_equal(
            x.device, ivy_backend.dev(data), as_array=False
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
    ),
)
def test_array_property_ndim(
    dtype_x,
    backend_fw,
):
    _, data, input_shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ivy_backend.utils.assertions.check_equal(
            x.ndim, len(input_shape), as_array=False
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
    ),
)
def test_array_property_shape(
    dtype_x,
    backend_fw,
):
    _, data, input_shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ivy_backend.utils.assertions.check_equal(
            x.shape, ivy_backend.Shape(input_shape), as_array=False
        )


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
    backend_fw,
):
    _, data, input_shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        size_gt = 1
        for dim in input_shape:
            size_gt *= dim
        ivy_backend.utils.assertions.check_equal(x.size, size_gt, as_array=False)


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_array_property_itemsize(
    dtype_x,
    backend_fw,
):
    dtype, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ivy_backend.utils.assertions.check_equal(
            x.itemsize, ivy_backend.to_numpy(x).itemsize, as_array=False
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_array_property_strides(dtype_x, backend_fw):
    dtype, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ivy_backend.utils.assertions.check_equal(
            x.strides, ivy_backend.to_numpy(x).strides, as_array=False
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
    ),
)
def test_array_property_mT(
    dtype_x,
    backend_fw,
    test_flags,
):
    _, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ret = helpers.flatten_and_to_np(ret=x.mT, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(
            ret=ivy_backend.matrix_transpose(data), backend=backend_fw
        )
        helpers.value_test(
            ret_np_flat=ret,
            ret_np_from_gt_flat=ret_gt,
            backend=backend_fw,
            ground_truth_backend=test_flags.ground_truth_backend,
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
    backend_fw,
    test_flags,
):
    _, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ret = helpers.flatten_and_to_np(ret=x.T, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(
            ret=ivy_backend.matrix_transpose(data), backend=backend_fw
        )
        helpers.value_test(
            ret_np_flat=ret,
            ret_np_from_gt_flat=ret_gt,
            backend=backend_fw,
            ground_truth_backend=test_flags.ground_truth_backend,
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("complex")),
)
def test_array_property_real(
    dtype_x,
    backend_fw,
    test_flags,
):
    _, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ret = helpers.flatten_and_to_np(ret=x.real, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(ret=ivy_backend.real(x), backend=backend_fw)
        helpers.value_test(
            ret_np_flat=ret,
            ret_np_from_gt_flat=ret_gt,
            backend=backend_fw,
            ground_truth_backend=test_flags.ground_truth_backend,
        )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("complex")),
)
def test_array_property_imag(
    dtype_x,
    backend_fw,
    test_flags,
):
    _, data = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        data = ivy_backend.native_array(data[0])
        x = ivy_backend.Array(data)
        ret = helpers.flatten_and_to_np(ret=x.imag, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(ret=ivy_backend.imag(x), backend=backend_fw)
        helpers.value_test(
            ret_np_flat=ret,
            ret_np_from_gt_flat=ret_gt,
            backend=backend_fw,
            ground_truth_backend=test_flags.ground_truth_backend,
        )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__getitem__",
    ground_truth_backend="numpy",
    dtypes_x_query=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_array__getitem__(
    dtypes_x_query,
    init_flags,
    method_flags,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    on_device,
):
    dtypes, x, query = dtypes_x_query
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x},
        init_input_dtypes=[dtypes[0]],
        method_input_dtypes=[*dtypes[1:]],
        method_all_as_kwargs_np={"query": query},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__setitem__",
    ground_truth_backend="numpy",
    dtypes_x_query_val=helpers.dtype_array_query_val(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    # ToDo: fix container method
    method_container_flags=st.just([False]),
)
def test_array__setitem__(
    dtypes_x_query_val,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtypes, x, query, val = dtypes_x_query_val
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x},
        init_input_dtypes=[dtypes[0]],
        method_input_dtypes=[*dtypes[1:]],
        method_all_as_kwargs_np={"query": query, "val": val},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__pos__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__pos__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__neg__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__pow__",
    dtype_and_x=pow_helper(),
)
def test_array__pow__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x

    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in input_dtype))

    # Make sure x2 isn't a float when x1 is integer
    assume(
        not (ivy.is_int_dtype(input_dtype[0] and ivy.is_float_dtype(input_dtype[1])))
    )

    # Make sure x2 is non-negative when both is integer
    if ivy.is_int_dtype(input_dtype[1]) and ivy.is_int_dtype(input_dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = not_too_close_to_zero(x[0])
    x[1] = not_too_close_to_zero(x[1])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[input_dtype[0]],
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={"power": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__rpow__",
    dtype_and_x=pow_helper(),
)
def test_array__rpow__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x

    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in input_dtype))

    # Make sure x2 isn't a float when x1 is integer
    assume(
        not (ivy.is_int_dtype(input_dtype[0] and ivy.is_float_dtype(input_dtype[1])))
    )

    # Make sure x2 is non-negative when both is integer
    if ivy.is_int_dtype(input_dtype[1]) and ivy.is_int_dtype(input_dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = not_too_close_to_zero(x[0])
    x[1] = not_too_close_to_zero(x[1])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[1]},
        init_input_dtypes=[input_dtype[1]],
        method_input_dtypes=[input_dtype[0]],
        method_all_as_kwargs_np={"power": x[0]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ipow__",
    dtype_and_x=pow_helper(),
    method_container_flags=st.just([False]),
)
def test_array__ipow__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x

    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in input_dtype))

    # Make sure x2 isn't a float when x1 is integer
    assume(
        not (ivy.is_int_dtype(input_dtype[0] and ivy.is_float_dtype(input_dtype[1])))
    )

    # Make sure x2 is non-negative when both is integer
    if ivy.is_int_dtype(input_dtype[1]) and ivy.is_int_dtype(input_dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = not_too_close_to_zero(x[0])
    x[1] = not_too_close_to_zero(x[1])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[input_dtype[0]],
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={"power": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__iadd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__iadd__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__isub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__isub__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__imul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__imul__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__imod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__imod__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__itruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__itruediv__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=3.0,
        small_abs_safety_factor=3.0,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__floordiv__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__rfloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=3.0,
        small_abs_safety_factor=3.0,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_array__rfloordiv__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ifloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=3.0,
        small_abs_safety_factor=3.0,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__ifloordiv__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__matmul__",
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
)
def test_array__matmul__(
    x,
    y,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype1, x = x
    input_dtype2, y = y
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x},
        init_input_dtypes=input_dtype1,
        method_input_dtypes=input_dtype2,
        method_all_as_kwargs_np={"other": y},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__rmatmul__",
    x1=_get_first_matrix_and_dtype(),
    x2=_get_second_matrix_and_dtype(),
)
def test_array__rmatmul__(
    x1,
    x2,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype1, x1 = x1
    dtype2, x2 = x2
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x2},
        init_input_dtypes=dtype1,
        method_input_dtypes=dtype2,
        method_all_as_kwargs_np={"other": x1},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__imatmul__",
    x1=_get_first_matrix_and_dtype(),
    x2=_get_second_matrix_and_dtype(),
    method_container_flags=st.just([False]),
)
def test_array__imatmul__(
    x1,
    x2,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype1, x1 = x1
    dtype2, x2 = x2
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x1},
        init_input_dtypes=dtype1,
        method_input_dtypes=dtype2,
        method_all_as_kwargs_np={"other": x2},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__abs__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=1.5,
        small_abs_safety_factor=1.5,
        safety_factor_scale="log",
    ),
)
def test_array__abs__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__float__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
    ),
)
def test_array__float__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__int__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
        min_value=-1e15,
        max_value=1e15,
    ),
    method_container_flags=st.just([False]),
)
def test_array__int__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__complex__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
    ),
    method_container_flags=st.just([False]),
)
def test_array__complex__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__lt__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__le__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__eq__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ne__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__gt__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ge__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__and__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__and__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__rand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__rand__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__iand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__iand__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__or__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ror__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__ror__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ior__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__ior__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
    ),
)
def test_array__invert__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__xor__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__rxor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_array__rxor__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ixor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__ixor__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__lshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__lshift__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    max_bits = np.iinfo(dtype[0]).bits
    max_shift = max_bits - 1
    x[1] = np.asarray(np.clip(x[1], 0, max_shift), dtype=dtype[1])
    max_value_before_shift = 2 ** (max_bits - x[1]) - 1
    overflow_threshold = 2 ** (max_bits - 1)
    x[0] = np.asarray(np.clip(x[0], None, max_value_before_shift), dtype=dtype[0])
    if np.any(x[0] > overflow_threshold):
        x[0] = np.asarray(np.clip(x[0], None, overflow_threshold), dtype=dtype[0])
    if np.any(x[0] < 0):
        x[0] = np.asarray(np.abs(x[0]), dtype=dtype[0])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[dtype[0]],
        method_input_dtypes=[dtype[1]],
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-5,
        atol_=1e-5,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__rlshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__rlshift__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    x[0] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[dtype[0]],
        method_input_dtypes=[dtype[1]],
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__ilshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__ilshift__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    x[1] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[dtype[0]],
        method_input_dtypes=[dtype[1]],
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    x[1] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[dtype[0]],
        method_input_dtypes=[dtype[1]],
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__rrshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
)
def test_array__rrshift__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    x[0] = np.asarray(np.clip(x[0], 0, np.iinfo(dtype[0]).bits - 1), dtype=dtype[0])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[dtype[0]],
        method_input_dtypes=[dtype[1]],
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__irshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        array_api_dtypes=True,
    ),
    method_container_flags=st.just([False]),
)
def test_array__irshift__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    x[1] = np.asarray(np.clip(x[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[dtype[0]],
        method_input_dtypes=[dtype[1]],
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__deepcopy__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_array__deepcopy__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
        method_all_as_kwargs_np={"memodict": {}},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__len__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_array__len__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Array.__iter__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=2,
        min_num_dims=1,
    ),
)
def test_array__iter__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        backend_to_test=backend_fw,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_test(
    fn_tree="functional.ivy.native_array",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=3,
        max_dim_size=3,
        min_num_dims=3,
        max_num_dims=3,
        num_arrays=2,
        min_value=3.0,
        max_value=10.0,
    ),
    op=st.sampled_from(
        ["!=", ">", "<", ">=", "<=", "*", "/", "%", "==", "&", "@", "**", "/"]
    ),
)
def test_dunder_wrapping(
    dtype_x,
    backend_fw,
    test_flags,
    op,
):
    _, data = dtype_x
    ivy.set_backend(backend_fw)
    x = ivy.to_native(ivy.array(data[0]))
    y = ivy.array(data[1])
    assert ivy.is_ivy_array(y)
    assert ivy.is_native_array(x)
    res = eval(f"x {op} y")
    assert ivy.is_ivy_array(res)
