# global
from hypothesis import strategies as st, assume
import numpy as np


# local
import ivy
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
    _get_castable_dtype,
)
from ivy import inf


# einsum
@handle_frontend_test(
    fn_tree="jax.numpy.einsum",
    eq_n_op=st.sampled_from(
        [
            (
                "ii",
                np.arange(25).reshape(5, 5),
            ),
            (
                "ii->i",
                np.arange(25).reshape(5, 5),
            ),
            ("ij,j", np.arange(25).reshape(5, 5), np.arange(5)),
        ]
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_numpy_einsum(
    *,
    eq_n_op,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    kw = {}
    i = 0
    for x_ in eq_n_op:
        kw["x{}".format(i)] = x_
        i += 1
    test_flags.num_positional_args = i
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **kw,
        out=None,
        optimize="optimal",
        precision=None,
        _use_xeinsum=False,
    )


# mean
@handle_frontend_test(
    fn_tree="jax.numpy.mean",
    dtype_x_axis=statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_mean(
    *,
    dtype_x_axis,
    dtype,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_helpers.test_frontend_function(
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
        keepdims=keepdims,
        where=where,
    )


# var
@handle_frontend_test(
    fn_tree="jax.numpy.var",
    dtype_x_axis=statistical_dtype_values(function="var").filter(
        lambda x: x[0][0] != "bfloat16"
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True).filter(
        lambda x: x != "bfloat16"
    ),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_var(
    *,
    dtype_x_axis,
    dtype,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis, ddof = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
        atol=1e-3,
        rtol=1e-3,
    )


# argmin
@handle_frontend_test(
    fn_tree="jax.numpy.argmin",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        valid_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_jax_numpy_argmin(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
    )


# bincount
@handle_frontend_test(
    fn_tree="jax.numpy.bincount",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=2,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            ),
            key="a_s_d",
        ),
    ),
    test_with_out=st.just(False),
)
def test_jax_numpy_bincount(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        weights=None,
        minlength=0,
        length=None,
    )


# cumprod
@handle_frontend_test(
    fn_tree="jax.numpy.cumprod",
    aliases=["jax.numpy.cumproduct"],
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", none=True, full=False),
    test_with_out=st.just(False),
)
def test_jax_numpy_cumprod(
    *,
    dtype_x_axis,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
    )


# cumsum
@handle_frontend_test(
    fn_tree="jax.numpy.cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("numeric", none=True, full=False),
    test_with_out=st.just(False),
)
def test_jax_numpy_cumsum(
    *,
    dtype_x_axis,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
    )


# sum
@handle_frontend_test(
    fn_tree="jax.numpy.sum",
    dtype_x_axis_castable=_get_castable_dtype(),
    initial=st.none() | st.floats(-10.0, 10.0),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_sum(
    *,
    dtype_x_axis_castable,
    initial,
    where,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis, castable_dtype = dtype_x_axis_castable

    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_helpers.test_frontend_function(
        input_dtypes=[input_dtypes],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-2,
        a=x[0],
        axis=axis,
        dtype=castable_dtype,
        out=None,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# min
@handle_frontend_test(
    fn_tree="jax.numpy.min",
    aliases=["jax.numpy.amin"],
    dtype_x_axis=statistical_dtype_values(function="min"),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_min(
    *,
    dtype_x_axis,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
    )


# max
@handle_frontend_test(
    fn_tree="jax.numpy.max",
    aliases=["jax.numpy.amax"],
    dtype_x_axis=statistical_dtype_values(function="max"),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_max(
    *,
    dtype_x_axis,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
    )


# average
@handle_frontend_test(
    fn_tree="jax.numpy.average",
    dtype_x_axis=helpers.dtype_values_axis(
        num_arrays=2,
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        valid_axis=True,
        allow_neg_axes=False,
        min_axes_size=1,
    ),
    returned=st.booleans(),
)
def test_jax_numpy_average(
    *,
    dtype_x_axis,
    returned,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtype, x, axis = dtype_x_axis

    if isinstance(axis, tuple):
        axis = axis[0]

    np_helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=2e-2,
        rtol=2e-2,
        a=x[0],
        axis=axis,
        weights=x[1],
        returned=returned,
    )


# nanmax
@handle_frontend_test(
    fn_tree="jax.numpy.nanmax",
    aliases=["jax.numpy.nanmax"],
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
    where=np_helpers.where(),
)
def test_numpy_nanmax(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is -inf)

    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
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
    fn_tree="jax.numpy.nanmin",
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
    where=np_helpers.where(),
)
def test_numpy_nanmin(
    dtype_x_axis,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    where,
    initial,
    keepdims,
):
    if initial is None and np.all(where) is not True:
        assume(initial is inf)

    input_dtypes, x, axis = dtype_x_axis
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
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


# nanstd
@handle_frontend_test(
    fn_tree="jax.numpy.nanstd",
    dtype_and_a=statistical_dtype_values(function="nanstd"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_jax_numpy_nanstd(
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


# nanvar
@handle_frontend_test(
    fn_tree="jax.numpy.nanvar",
    dtype_x_axis=statistical_dtype_values(function="nanvar").filter(
        lambda x: x[0][0] != "bfloat16"
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True).filter(
        lambda x: x != "bfloat16"
    ),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_nanvar(
    *,
    dtype_x_axis,
    dtype,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis, ddof = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    assume(np.dtype(dtype[0]) >= np.dtype(input_dtypes[0]))
    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
        atol=1e-3,
        rtol=1e-3,
    )


@st.composite
def _get_castable_dtypes_values(draw, *, allow_nan=False, use_where=False):
    available_dtypes = helpers.get_dtypes("numeric")
    shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=4, max_dim_size=6))
    dtype, values = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            num_arrays=1,
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            shape=shape,
            allow_nan=allow_nan,
        )
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    dtype1, values, dtype2 = draw(
        helpers.get_castable_dtype(draw(available_dtypes), dtype[0], values[0])
    )
    if use_where:
        where = draw(np_frontend_helpers.where(shape=shape))
        return [dtype1], [values], axis, dtype2, where
    return [dtype1], [values], axis, dtype2


# nancumprod
@handle_frontend_test(
    fn_tree="jax.numpy.nancumprod",
    dtype_and_x_axis_dtype=_get_castable_dtypes_values(allow_nan=True),
)
def test_jax_numpy_nancumprod(
    dtype_and_x_axis_dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, axis, dtype = dtype_and_x_axis_dtype
    if ivy.current_backend_str() == "torch":
        assume(not test_flags.as_variable[0])
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype,
    )


# nancumsum
@handle_frontend_test(
    fn_tree="jax.numpy.nancumsum",
    dtype_and_x_axis_dtype=_get_castable_dtypes_values(allow_nan=True),
)
def test_jax_numpy_nancumsum(
    dtype_and_x_axis_dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, axis, dtype = dtype_and_x_axis_dtype
    if ivy.current_backend_str() == "torch":
        assume(not test_flags.as_variable[0])
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype,
    )


# std
@handle_frontend_test(
    fn_tree="jax.numpy.std",
    dtype_x_axis=statistical_dtype_values(function="std"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_helpers.where(),
    keepdims=st.booleans(),
)
def test_jax_numpy_std(
    *,
    dtype_x_axis,
    dtype,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis, ddof = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
        atol=1e-3,
        rtol=1e-3,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.corrcoef",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=2,
        shared_dtype=True,
        abs_smallest_val=1e-5,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    rowvar=st.booleans(),
)
def test_jax_numpy_corrcoef(
    dtype_and_x,
    rowvar,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x = dtype_and_x
    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
        rowvar=rowvar,
    )


# median
@handle_frontend_test(
    fn_tree="jax.numpy.median",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_value=-(2**10),
        max_value=2**10,
        valid_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_jax_numpy_median(
    *,
    dtype_x_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        overwrite_input=False,
        keepdims=keepdims,
        atol=1e-3,
        rtol=1e-3,
    )


# ptp
@handle_frontend_test(
    fn_tree="jax.numpy.ptp",
    dtype_and_x_axis_dtype=_get_castable_dtypes_values(allow_nan=False),
    keep_dims=st.booleans(),
)
def test_jax_numpy_ptp(
    dtype_and_x_axis_dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, x, axis, dtype = dtype_and_x_axis_dtype
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keep_dims,
    )


# nanmean
@st.composite
def _get_castable_dtype_with_nan(draw):
    available_dtypes = helpers.get_dtypes("float")
    shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=4, max_dim_size=6))
    dtype, values = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            num_arrays=1,
            large_abs_safety_factor=6,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            shape=shape,
            allow_nan=True,
            allow_inf=True,
        )
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    dtype1, values, dtype2 = draw(
        helpers.get_castable_dtype(draw(available_dtypes), dtype[0], values[0])
    )
    return dtype1, [values], axis, dtype2


@handle_frontend_test(
    fn_tree="jax.numpy.nanmean",
    dtype_x_axis_castable_dtype=_get_castable_dtype_with_nan(),
    keepdims=st.booleans(),
    where=np_helpers.where(),
)
def test_jax_numpy_nanmean(
    dtype_x_axis_castable_dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    where,
    keepdims,
):
    input_dtypes, x, axis, castable_dtype = dtype_x_axis_castable_dtype
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=[input_dtypes],
        test_flags=test_flags,
    )
    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        a=x[0],
        axis=axis,
        dtype=castable_dtype,
        out=None,
        keepdims=keepdims,
        where=where,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.nanmedian",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        min_value=-(2**10),
        max_value=2**10,
        valid_axis=True,
    ),
    keepdims=st.booleans(),
    overwrite_input=st.booleans(),
)
def test_jax_numpy_nanmedian(
    on_device,
    frontend,
    *,
    dtype_x_axis,
    keepdims,
    fn_tree,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        overwrite_input=False,
        keepdims=keepdims,
    )
