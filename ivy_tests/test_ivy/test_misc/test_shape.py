from hypothesis import strategies as st

import ivy
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_method


CLASS_TREE = "ivy.Shape"
DUMMY_DTYPE = ["int32"]


# --- Helpers --- #
# --------------- #


@st.composite
def shape_and_index(draw):
    shape = draw(helpers.get_shape())
    index = draw(
        st.integers(min_value=0, max_value=len(shape) - 1)
        if len(shape) > 0
        else st.just(0)
    )
    return shape, index


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
    shape=helpers.get_shape(min_num_dims=0),
)
def test_shape__bool__(
    shape,
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
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Shape.__eq__",
    shape_1=helpers.get_shape(),
    shape_2=helpers.get_shape(),
)
def test_shape__eq__(
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
    init_tree=CLASS_TREE,
    method_tree="Shape.__getitem__",
    shape_index=shape_and_index(),
)
def test_shape__getitem__(
    shape_index,
    init_flags,
    method_flags,
    method_name,
    class_name,
    backend_fw,
    ground_truth_backend,
    on_device,
):
    shape, query = shape_index
    helpers.test_method(
        on_device=on_device,
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={"shape_tup": shape},
        init_input_dtypes=DUMMY_DTYPE,
        method_input_dtypes=DUMMY_DTYPE,
        method_all_as_kwargs_np={"key": query},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Shape.__gt__",
    shape_1=helpers.get_shape(),
    shape_2=helpers.get_shape(),
)
def test_shape__gt__(
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
    method_tree="Shape.__iter__",
    shape=helpers.get_shape(),
)
def test_shape__iter__(
    shape,
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
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Shape.__le__",
    shape_1=helpers.get_shape(),
    shape_2=helpers.get_shape(),
)
def test_shape__le__(
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
    method_tree="Shape.__len__",
    shape=helpers.get_shape(),
)
def test_shape__len__(
    shape,
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
        method_all_as_kwargs_np={},
        class_name=class_name,
        method_name=method_name,
    )


@handle_method(
    init_tree=CLASS_TREE,
    method_tree="Shape.__lt__",
    shape_1=helpers.get_shape(),
    shape_2=helpers.get_shape(),
)
def test_shape__lt__(
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


def test_shape_in_conditions():
    shape = ivy.Shape((1, 2))
    condition_is_true = True if shape else False
    assert condition_is_true

    shape = ivy.Shape(())
    condition_is_true = True if shape else False
    assert not condition_is_true
