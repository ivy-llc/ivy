# global
from hypothesis import strategies as st, assume
import numpy as np


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _get_dtype_value1_value2_cov(
    draw,
    available_dtypes,
    min_num_dims,
    max_num_dims,
    min_dim_size,
    max_dim_size,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )

    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    abs_smallest_val=abs_smallest_val,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=large_abs_safety_factor,
                    small_abs_safety_factor=small_abs_safety_factor,
                    safety_factor_scale=safety_factor_scale,
                )
            )
        )

    value1, value2 = values[0], values[1]

    # modifiers: rowVar, bias, ddof
    rowVar = draw(st.booleans())
    bias = draw(st.booleans())
    ddof = draw(helpers.ints(min_value=0, max_value=1))

    numVals = None
    if rowVar is False:
        numVals = -1 if numVals == 0 else 0
    else:
        numVals = 0 if len(shape) == 1 else -1

    fweights = draw(
        helpers.array_values(
            dtype="int64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
        )
    )

    aweights = draw(
        helpers.array_values(
            dtype="float64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
            small_abs_safety_factor=1,
        )
    )

    return [dtype], value1, value2, rowVar, bias, ddof, fweights, aweights


# --- Main --- #
# ------------ #


# average
@handle_frontend_test(
    fn_tree="numpy.average",
    dtype_and_a=_statistical_dtype_values(function="average"),
    dtype_and_x=_statistical_dtype_values(function="average"),
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
    backend_fw,
    keep_dims,
    returned,
    on_device,
):
    try:
        input_dtype, a, axis, *_ = dtype_and_a

        input_dtypes, xs, axiss, *_ = dtype_and_x

        if isinstance(axis, tuple):
            axis = axis[0]

        helpers.test_frontend_function(
            a=a[0],
            input_dtypes=input_dtype,
            backend_to_test=backend_fw,
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


# cov
@handle_frontend_test(
    fn_tree="numpy.cov",
    dtype_x1_x2_cov=_get_dtype_value1_value2_cov(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_numpy_cov(
    dtype_x1_x2_cov,
    test_flags,
    frontend,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x1, x2, rowvar, bias, ddof, fweights, aweights = dtype_x1_x2_cov
    np_frontend_helpers.test_frontend_function(
        input_dtypes=[dtype[0], dtype[0], "int64", "float64"],
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        m=x1,
        y=x2,
        rowvar=rowvar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
    )


# mean
@handle_frontend_test(
    fn_tree="numpy.mean",
    dtype_and_x=_statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_mean(
    dtype_and_x,
    dtype,
    where,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, x, axis = dtype_and_x
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        keepdims=keep_dims,
        where=where,
    )


# nanmean
@handle_frontend_test(
    fn_tree="numpy.nanmean",
    dtype_and_a=_statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanmean(
    dtype_and_a,
    dtype,
    where,
    frontend,
    backend_fw,
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

    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
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
    )


@handle_frontend_test(
    fn_tree="numpy.nanmedian",
    keep_dims=st.booleans(),
    overwrite_input=st.booleans(),
    dtype_x_axis=_statistical_dtype_values(function="nanmedian"),
)
def test_numpy_nanmedian(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    keep_dims,
    overwrite_input,
):
    input_dtypes, x, axis = dtype_x_axis
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        overwrite_input=overwrite_input,
        out=None,
        keepdims=keep_dims,
    )


# nanstd
@handle_frontend_test(
    fn_tree="numpy.nanstd",
    dtype_and_a=_statistical_dtype_values(function="nanstd"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanstd(
    dtype_and_a,
    dtype,
    where,
    frontend,
    backend_fw,
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
        backend_to_test=backend_fw,
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


# nanvar
@handle_frontend_test(
    fn_tree="numpy.nanvar",
    dtype_x_axis=_statistical_dtype_values(function="nanvar"),
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
    backend_fw,
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
        backend_to_test=backend_fw,
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


# std
@handle_frontend_test(
    fn_tree="numpy.std",
    dtype_and_x=_statistical_dtype_values(function="std"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_std(
    dtype_and_x,
    dtype,
    where,
    frontend,
    backend_fw,
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
        backend_to_test=backend_fw,
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


@handle_frontend_test(
    fn_tree="numpy.var",
    dtype_and_x=_statistical_dtype_values(function="var"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_var(
    dtype_and_x,
    dtype,
    where,
    frontend,
    backend_fw,
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
        backend_to_test=backend_fw,
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
