import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method

CLASS_TREE = "ivy.functional.frontends.xgboost.core.DMatrix"


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="xgboost.DMatrix",
    method_name="num_col",
    init_array=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="valid"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=10,
    ),
)
def test_xgboost_instance_num_col(
    init_array,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    dtype, arr = init_array
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": arr[0]},
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="xgboost.DMatrix",
    method_name="num_row",
    init_array=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="valid"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=10,
    ),
)
def test_xgboost_instance_num_row(
    init_array,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    dtype, arr = init_array
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": arr[0]},
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )
