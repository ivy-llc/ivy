import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method


CLASS_TREE = "ivy.functional.frontends.sklearn.preprocessing"


@handle_frontend_method(
    class_tree=CLASS_TREE + ".LabelEncoder",
    init_tree="sklearn.preprocessing.LabelEncoder",
    method_name="fit",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_sklearn_label_encoder_fit(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE + ".LabelEncoder",
    init_tree="sklearn.preprocessing.LabelEncoder",
    method_name="fit_transform",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_sklearn_label_encoder_fit_transform(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    # TODO: add test for this method with encoder already fitted with data
    pass


@handle_frontend_method(
    class_tree=CLASS_TREE + ".LabelEncoder",
    init_tree="sklearn.preprocessing.LabelEncoder",
    method_name="inverse_transform",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_sklearn_label_encoder_inverse_transform(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    # TODO: add test for this method with encoder already fitted with data
    pass


@handle_frontend_method(
    class_tree=CLASS_TREE + ".LabelEncoder",
    init_tree="sklearn.preprocessing.LabelEncoder",
    method_name="transform",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_sklearn_label_encoder_transform(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    # TODO: add test for this method with encoder already fitted with data
    pass
