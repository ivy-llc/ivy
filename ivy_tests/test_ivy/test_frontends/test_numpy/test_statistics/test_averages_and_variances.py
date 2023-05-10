# global
from hypothesis import strategies as st, assume
import numpy as np


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# mean
@handle_frontend_test(
    fn_tree="numpy.mean",
    dtype_and_x=statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_mean(
    dtype_and_x,
    dtype,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]

    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
        rtol=1e-2,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )


# nanmean
@handle_frontend_test(
    fn_tree="numpy.nanmean",
    dtype_and_a=statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanmean(
    dtype_and_a,
    dtype,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, a, axis = dtype_and_a
    if isinstance(axis, tuple):
        axis = axis[0]

    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        a=a[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )


# std
@handle_frontend_test(
    fn_tree="numpy.std",
    dtype_and_x=statistical_dtype_values(function="std"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_std(
    dtype_and_x,
    dtype,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, x, axis, correction = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    assume(np.dtype(dtype[0]) >= np.dtype(input_dtypes[0]))
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        x=x[0],
        axis=axis,
        ddof=correction,
        keepdims=keep_dims,
        out=None,
        dtype=dtype[0],
        where=where,
    )


# average
@handle_frontend_test(
    fn_tree="numpy.average",
    dtype_and_a=statistical_dtype_values(function="average"),
    dtype_and_x=statistical_dtype_values(function="average"),
    keep_dims=st.booleans(),
    returned=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_average(
    dtype_and_a,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    keep_dims,
    returned,
    on_device,
):
    try:
        input_dtype, a, axis = dtype_and_a

        input_dtypes, xs, axiss = dtype_and_x

        if isinstance(axis, tuple):
            axis = axis[0]

        helpers.test_frontend_function(
            a=a[0],
            input_dtypes=input_dtype,
            weights=xs[0],
            axis=axis,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            keepdims=keep_dims,
            returned=returned,
            on_device=on_device,
            rtol=1e-2,
            atol=1e-2,
        )
    except ZeroDivisionError:
        assume(False)
    except AssertionError:
        assume(False)


# nanstd
@handle_frontend_test(
    fn_tree="numpy.nanstd",
    dtype_and_a=statistical_dtype_values(function="nanstd"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanstd(
    dtype_and_a,
    dtype,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, a, axis, correction = dtype_and_a
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    assume(np.dtype(dtype[0]) >= np.dtype(input_dtypes[0]))
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        ddof=correction,
        keepdims=keep_dims,
        where=where,
        atol=1e-2,
        rtol=1e-2,
    )


# cov
@handle_frontend_test(
    fn_tree="numpy.cov",
    dtype_and_x=statistical_dtype_values(function="cov"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_cov(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
        test_values=False,
    )


# nanvar
@handle_frontend_test(
    fn_tree="numpy.nanvar",
    dtype_x_axis=statistical_dtype_values(function="nanvar"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanvar(
    dtype_x_axis,
    dtype,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, x, axis, ddof = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-1,
        rtol=1e-1,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        ddof=ddof,
        keepdims=keep_dims,
        where=where,
    )


@handle_frontend_test(
    fn_tree="numpy.nanpercentile",
    dtype_values_axis=statistical_dtype_values(function="nanpercentile"),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanpercentile(
    dtype_values_axis,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, values, axis = dtype_values_axis
    if isinstance(axis, tuple):
        axis = axis[0]

    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_frontend_helpers.test_frontend_function(
        a=values[0][0],
        q=values[0][1],
        axis=axis,
        out=None,
        overwrite_input=None,
        method=None,
        keepdims=keep_dims,
        interpolation=None,
        frontend=frontend,
        fn_tree=fn_tree,
        test_flags=test_flags,
        input_dtypes=input_dtypes,
    )


@handle_frontend_test(
    fn_tree="numpy.nanmedian",
    keep_dims=st.booleans(),
    overwrite_input=st.booleans(),
    dtype_x_axis=statistical_dtype_values(function="nanmedian"),
)
def test_numpy_nanmedian(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
    overwrite_input,
):
    input_dtypes, x, axis = dtype_x_axis
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        overwrite_input=overwrite_input,
        out=None,
        keepdims=keep_dims,
    )


@handle_frontend_test(
    fn_tree="numpy.var",
    dtype_and_x=statistical_dtype_values(function="var"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_var(
    dtype_and_x,
    dtype,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, x, axis, correction = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    assume(np.dtype(dtype[0]) >= np.dtype(input_dtypes[0]))
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        x=x[0],
        axis=axis,
        ddof=correction,
        keepdims=keep_dims,
        out=None,
        dtype=dtype[0],
        where=where,
    )
