# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


@handle_frontend_test(
    fn_tree="numpy.isneginf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_numpy_isneginf(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.isposinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_numpy_isposinf(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.allclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
    equal_nan=st.booleans(),
)
def test_numpy_allclose(
    *,
    dtype_and_x,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-05,
        atol=1e-08,
        a=x[0],
        b=x[1],
        equal_nan=equal_nan,
    )


@handle_frontend_test(
    fn_tree="numpy.isclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
    equal_nan=st.booleans(),
)
def test_numpy_isclose(
    *,
    dtype_and_x,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-05,
        atol=1e-08,
        a=x[0],
        b=x[1],
        equal_nan=equal_nan,
    )


@handle_frontend_test(
    fn_tree="numpy.isnat",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    dtype=helpers.get_dtypes("float"),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.isnat"
    ),
)
def test_numpy_isnat(
    *,
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=[input_dtype],
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="isnat",
        x=np.asarray(x, dtype=input_dtype),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        subok=True,
        test_values=False,
    )
