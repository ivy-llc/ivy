# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method
import pytest

CLASS_TREE = "ivy.functional.frontends.tensorflow.tensor.TensorShape"


# __add__
@pytest.mark.skip("TODO: test needs implementing correctly")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.TensorShape",
    method_name="__add__",
    shape_list=helpers.list_of_size(x=st.sampled_from([0, 1, 2, 3, 4]), size=3),
    other_list=helpers.list_of_size(x=st.sampled_from([0, 1, 2, 3, 4]), size=3),
)
def test_tensorflow__add__(
    shape_list,
    other_list,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    helpers.test_frontend_method(
        init_input_dtypes=[ivy.int64],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "dims": shape_list,
        },
        method_input_dtypes=[ivy.int64],
        method_all_as_kwargs_np={
            "other": other_list,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __bool__
@pytest.mark.skip("TODO: test needs implementing correctly")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="tensorflow.TensorShape",
    method_name="__bool__",
    shape_list=helpers.list_of_size(x=st.sampled_from([0, 1, 2, 3, 4]), size=3),
)
def test_tensorflow__bool__(
    shape_list,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    on_device,
):
    helpers.test_frontend_method(
        init_input_dtypes=[ivy.int64],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "dims": shape_list,
        },
        method_input_dtypes=[ivy.int64],
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )
