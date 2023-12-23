# global
import numpy as np
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy import inf


# amax
@handle_frontend_test(
    fn_tree="numpy.amax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    initial=st.one_of(st.floats(min_value=-1000, max_value=1000), st.none()),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_amax(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is +inf)
    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# amin
@handle_frontend_test(
    fn_tree="numpy.amin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    initial=st.one_of(st.floats(min_value=-1000, max_value=1000), st.none()),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_amin(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is inf)

    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# fmax
@handle_frontend_test(
    fn_tree="numpy.fmax",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
        shared_dtype=True,
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="fmax"
    ),
)
def test_numpy_fmax(
    dtype_and_inputs,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs = dtype_and_inputs
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
    )


# fmin
@handle_frontend_test(
    fn_tree="numpy.fmin",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
                num_arrays=2,
                shared_dtype=True,
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="fmin"
    ),
)
def test_numpy_fmin(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


@handle_frontend_test(
    fn_tree="numpy.max",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    initial=st.one_of(st.floats(min_value=-1000, max_value=1000), st.none()),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_max(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is +inf)
    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# maximum
@handle_frontend_test(
    fn_tree="numpy.maximum",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
                num_arrays=2,
                shared_dtype=True,
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="maximum"
    ),
)
def test_numpy_maximum(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


@handle_frontend_test(
    fn_tree="numpy.min",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    initial=st.one_of(st.floats(min_value=-1000, max_value=1000), st.none()),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_min(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is inf)

    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# minimum
@handle_frontend_test(
    fn_tree="numpy.minimum",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
                num_arrays=2,
                shared_dtype=True,
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="minimum"
    ),
)
def test_numpy_minimum(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# nanmax
@handle_frontend_test(
    fn_tree="numpy.nanmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
        allow_nan=True,
        allow_inf=True,
    ),
    initial=st.one_of(st.floats(min_value=-1000, max_value=1000), st.none()),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_nanmax(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is -inf)

    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# nanmin
@handle_frontend_test(
    fn_tree="numpy.nanmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
        allow_nan=True,
        allow_inf=True,
    ),
    initial=st.one_of(st.floats(min_value=-1000, max_value=1000), st.none()),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_nanmin(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is inf)

    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )
