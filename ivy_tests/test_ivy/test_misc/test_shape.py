from hypothesis import assume, strategies as st
import numpy as np

import ivy
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_method


CLASS_TREE = "ivy.Shape"
DUMMY_DTYPE = ["int32"]


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Shape.__add__",
    shape_1=helpers.get_shape(),
    shape_2=helpers.get_shape(),
)
def test_shape__add__(
    shape_1,
    shape_2,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"shape_tup": shape_1},
        init_input_dtypes=DUMMY_DTYPE,
        method_input_dtypes=DUMMY_DTYPE,
        method_all_as_kwargs_np={"other": shape_2},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
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
        init_all_as_kwargs_np={"shape_tup": x},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
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
    init_tree=CLASS_TREE,
    method_tree="Shape.__ge__",
    shape_1=helpers.get_shape(),
    shape_2=helpers.get_shape(),
)
def test_shape__ge__(
    shape_1,
    shape_2,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"shape_tup": shape_1},
        init_input_dtypes=DUMMY_DTYPE,
        method_input_dtypes=DUMMY_DTYPE,
        method_all_as_kwargs_np={"other": shape_2},
        class_name=class_name,
        method_name=method_name,
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
        init_all_as_kwargs_np={"shape": x[0]},
        init_input_dtypes=[dtypes[0]],
        method_input_dtypes=[dtypes[1]],
        method_all_as_kwargs_np={"key": query},
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
        init_all_as_kwargs_np={"shape": x[0]},
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
        init_all_as_kwargs_np={"shape": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=[],
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
        init_all_as_kwargs_np={"shape": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
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
        init_all_as_kwargs_np={"shape": x[0]},
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
    init_tree=CLASS_TREE,
    method_tree="Shape.__mul__",
    shape=helpers.get_shape(),
    other=st.integers(min_value=1, max_value=10),
)
def test_shape__mul__(
    shape,
    other,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"shape_tup": shape},
        init_input_dtypes=DUMMY_DTYPE,
        method_input_dtypes=DUMMY_DTYPE,
        method_all_as_kwargs_np={"other": other},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Shape.__radd__",
    shape_1=helpers.get_shape(),
    shape_2=helpers.get_shape(),
)
def test_shape__radd__(
    shape_1,
    shape_2,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"shape_tup": shape_1},
        init_input_dtypes=DUMMY_DTYPE,
        method_input_dtypes=DUMMY_DTYPE,
        method_all_as_kwargs_np={"other": shape_2},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    method_tree="Shape.__rdiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_shape__rdiv__(
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
        init_all_as_kwargs_np={"shape": x[0]},
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
    init_tree=CLASS_TREE,
    method_tree="Shape.__rmul__",
    shape=helpers.get_shape(),
    other=st.integers(min_value=1, max_value=10),
)
def test_shape__rmul__(
    shape,
    other,
    method_name,
    class_name,
    ground_truth_backend,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    helpers.test_method(
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"shape_tup": shape},
        init_input_dtypes=DUMMY_DTYPE,
        method_input_dtypes=DUMMY_DTYPE,
        method_all_as_kwargs_np={"other": other},
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
        init_all_as_kwargs_np={"shape": x[0]},
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
        init_all_as_kwargs_np={"shape": x[0]},
        init_input_dtypes=dtype,
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": x[1]},
        class_name=class_name,
        method_name=method_name,
    )


def test_shape_in_conditions():
    shape = ivy.Shape((1, 2))
    condition_is_true = True if shape else False
    assert condition_is_true

    shape = ivy.Shape(())
    condition_is_true = True if shape else False
    assert not condition_is_true
