from hypothesis import assume, strategies as st
import numpy as np

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method, BackendHandler
from ivy_tests.test_ivy.test_functional.test_core.test_elementwise import (
    not_too_close_to_zero,
    pow_helper,
)


@handle_method(
    method_tree="Shape.__getitem__",
    dtypes_x_query=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ),
)
def test_shape__getitem__(
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
        on_device=on_device,
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"shape_tup": x},
        init_input_dtypes=[dtypes[0]],
        method_input_dtypes=[dtypes[1]],
        method_all_as_kwargs_np={"key": query},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Shape.__index__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_shape__index__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__pow__",
    dtype_and_x=pow_helper(),
)
def test_shape__pow__(
    dtype_and_x,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x

    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in input_dtype))

    # Make sure x2 isn't a float when x1 is integer
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        assume(
            not (
                ivy_backend.is_int_dtype(
                    input_dtype[0] and ivy_backend.is_float_dtype(input_dtype[1])
                )
            )
        )

        # Make sure x2 is non-negative when both is integer
        if ivy_backend.is_int_dtype(input_dtype[1]) and ivy_backend.is_int_dtype(
            input_dtype[0]
        ):
            x[1] = np.abs(x[1])

        x[0] = not_too_close_to_zero(x[0])
        x[1] = not_too_close_to_zero(x[1])

    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        backend_to_test=backend_fw,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=[input_dtype[0]],
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={"power": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Shape.__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__add__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__radd__(
    dtype_and_x,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__sub__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__rsub__(
    dtype_and_x,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__mul__(
    dtype_and_x,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
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
    method_tree="Shape.__rmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__rmul__(
    dtype_and_x,
    method_name,
    class_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
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
    method_tree="Shape.__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__mod__(
    dtype_and_x,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__rmod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__rmod__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__rdivmod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__rdivmod__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__truediv__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__rtruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__rtruediv__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__rfloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=3.0,
        small_abs_safety_factor=3.0,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__rfloordiv__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__int__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
        min_value=-1e15,
        max_value=1e15,
    ),
    method_container_flags=st.just([False]),
)
def test_shape__int__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__bool__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        max_num_dims=0,
        min_value=0,
        max_value=1,
    ),
)
def test_shape__bool__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_shape__lt__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_shape__le__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_shape__eq__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_shape__ne__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_shape__gt__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_shape__ge__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__len__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=2,
        min_num_dims=1,
    ),
)
def test_shape__len__(
    dtype_and_x,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
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
    method_tree="Shape.__iter__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_dim_size=2,
        min_num_dims=1,
    ),
)
def test_shape__iter__(
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
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"data": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )
