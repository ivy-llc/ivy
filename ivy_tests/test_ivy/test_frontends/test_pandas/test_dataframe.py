# global
from hypothesis import strategies as st
import pytest
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method


CLASS_TREE = "ivy.functional.frontends.pandas.DataFrame"


@pytest.mark.skip("Testing pipeline not yet implemented")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="pandas.DataFrame",
    method_name="abs",
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_pandas_series_abs(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    on_device,
):
    # todo add castable dtypes for output
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@pytest.mark.xfail(reason="testing pipeline fixes")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="pandas.DataFrame",
    method_name="mean",
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    skipna=st.booleans(),
    axis=st.sampled_from([None, 0, 1, "index", "columns"]),
)
def test_pandas_series_mean(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
    skipna,
    axis,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"skipna": skipna, "axis": axis},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


@pytest.mark.xfail(reason="testing pipeline fixes")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="pandas.DataFrame",
    method_name="sum",
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    skipna=st.booleans(),
    axis=st.sampled_from([None, 0, 1, "index", "columns"]),
    min_count=st.integers(min_value=0, max_value=5),
)
def test_pandas_series_sum(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
    skipna,
    axis,
    min_count,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "skipna": skipna,
            "axis": axis,
            "min_count": min_count,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


@pytest.mark.skip("Testing pipeline not yet implemented")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="pandas.DataFrame",
    method_name="to_numpy",
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    na_values=st.sampled_from([None, np.nan, np.inf, -np.inf]),
    copy=st.booleans(),
)
def test_pandas_series_to_numpy(
    dtype_x,
    frontend,
    na_values,
    copy,
    frontend_method_data,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "na_values": na_values,
            "copy": copy,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )
